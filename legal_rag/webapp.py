"""Local document workspace with durable semantic ingestion and grounded chat."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from legal_rag.agent import research
from legal_rag.pipeline import Ingestion
from legal_rag.retrieval import Models, Retrieval, VectorStore
from legal_rag.settings import Settings
from legal_rag.storage import Repository

WEB = Path(__file__).resolve().parent / "web"


class Service:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.repository = Repository(settings)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.models = None
        self.vectors = VectorStore(settings.root)
        self.scheduled = set()
        self.lock = threading.Lock()

    def get_models(self) -> Models:
        if self.models is None:
            import torch

            torch.set_num_threads(2)
            self.models = Models(self.settings)
        return self.models

    def recover(self):
        for job in self.repository.jobs():
            if job["status"] in {"queued", "running"}:
                self.schedule(job)

    def schedule(self, job: dict):
        with self.lock:
            if job["id"] in self.scheduled:
                return
            self.scheduled.add(job["id"])
        self.worker.submit(self.run, job["id"])

    def run(self, job_id: str):
        job = self.repository.job(job_id)
        if job["status"] == "done":
            return
        self.repository.update_job(job_id, status="running", message="Preparing models")
        try:
            if job["kind"] == "ingest":
                result = Ingestion(
                    self.settings, self.repository, self.get_models(), self.vectors
                ).run(job)
            else:
                result = self.answer(job)
            self.repository.update_job(
                job_id, status="done", message="Complete", result=result
            )
        except Exception as error:
            logging.exception("Job %s failed", job_id)
            self.repository.update_job(job_id, status="error", message=str(error))
            if job["kind"] == "ingest":
                document = self.repository.document(job["document_id"])
                self.repository.update_document(
                    document["id"],
                    status="ready" if document.get("active_version") else "error",
                    message=f"Indexing failed: {error}",
                )
        finally:
            with self.lock:
                self.scheduled.discard(job_id)

    def answer(self, job: dict) -> dict:
        document = self.repository.document(job["document_id"])
        version = job["payload"]["version"]
        document["active_version"] = version
        folder = self.repository.version_folder(document["id"], version)
        retrieval = Retrieval(self.get_models(), self.vectors, document, folder)
        payload = job["payload"]
        result = research(
            payload["question"],
            retrieval,
            self.settings,
            lambda message: self.repository.update_job(job["id"], message=message),
            use_web=payload.get("use_web", False),
            history=payload.get("history", []),
        )
        evidence = []
        for item in result["evidence"]:
            parent = result["parents"][item["source"] - 1]
            atom_id = parent["atom_ids"][item["passage"] - 1]
            atom = retrieval.graph.nodes.get(atom_id, {})
            spans = atom.get("spans", parent.get("spans", []))
            page = next((span["page"] for span in spans if span.get("page")), 1)
            evidence.append(
                {
                    **item,
                    "title": parent["title"],
                    "page": page,
                    "pdf_url": f"/api/files/{document['id']}"
                    if document["suffix"] == ".pdf" and not parent.get("web")
                    else None,
                    "url": parent.get("url"),
                    "document_id": document["id"],
                    "version": version,
                    "preview_url": f"/api/preview/{document['id']}?version={version}#source-{spans[0]['start'] if spans else 0}",
                    "spans": spans,
                }
            )
        if result["mindmap"]:
            for node in result["mindmap"]["nodes"]:
                parent = result["parents"][node["source"] - 1]
                page = next(
                    (
                        span["page"]
                        for span in parent.get("spans", [])
                        if span.get("page")
                    ),
                    1,
                )
                node.update(
                    page=page,
                    pdf_url=f"/api/files/{document['id']}"
                    if document["suffix"] == ".pdf" and not parent.get("web")
                    else None,
                    url=parent.get("url"),
                    preview_url=f"/api/preview/{document['id']}?version={version}",
                )
        return {
            key: value
            for key, value in {
                **result,
                "evidence": evidence,
                "version": version,
            }.items()
            if key != "parents"
        }

    def close(self):
        self.worker.shutdown(wait=True)
        self.vectors.close()


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=2000)
    use_web: bool = False
    history: list[Turn] = Field(default_factory=list, max_length=4)


def create_app(root: Path | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if root is not None:
        settings = settings.model_copy(update={"root": root})
    service = Service(settings)

    @asynccontextmanager
    async def lifespan(app):
        service.recover()
        yield
        service.close()

    app = FastAPI(
        title="Semantic Document RAG", docs_url=None, redoc_url=None, lifespan=lifespan
    )
    app.state.service = service
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    def document_or_404(document_id: str):
        try:
            return service.repository.document(document_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/")
    def home():
        return FileResponse(WEB / "index.html")

    @app.get("/api/documents")
    def documents():
        return service.repository.documents()

    @app.get("/api/files/{document_id}")
    def original(document_id: str):
        document = document_or_404(document_id)
        return FileResponse(
            service.repository.source(document_id),
            filename=document["name"],
            media_type="application/pdf"
            if document["suffix"] == ".pdf"
            else "application/octet-stream",
            content_disposition_type="inline"
            if document["suffix"] == ".pdf"
            else "attachment",
        )

    @app.get("/api/preview/{document_id}")
    def preview(document_id: str, version: str | None = None):
        document = document_or_404(document_id)
        version = (
            version or document.get("pending_version") or document.get("active_version")
        )
        try:
            path = (
                service.repository.version_folder(document_id, version) / "preview.html"
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        if not path.exists():
            raise HTTPException(409, "The document preview is not available yet.")
        return FileResponse(
            path,
            media_type="text/html",
            headers={
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'self'"
            },
        )

    @app.get("/api/documents/{document_id}/graph")
    def graph(document_id: str):
        document_or_404(document_id)
        try:
            folder = service.repository.version_folder(document_id)
            return {
                "graph": json.loads((folder / "graph.json").read_text()),
                "plan": json.loads((folder / "plan.json").read_text()),
            }
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(
                409, "The document has no active semantic graph."
            ) from error

    @app.post("/api/documents")
    async def upload(file: UploadFile):
        content = await file.read(service.settings.max_upload_bytes + 1)
        try:
            document, needs_index = service.repository.add(
                file.filename or "document.pdf", content
            )
        except (ValueError, UnicodeError) as error:
            raise HTTPException(400, str(error)) from error
        job = (
            service.repository.submit(document["id"], "ingest") if needs_index else None
        )
        if job:
            service.repository.update_document(
                document["id"], status="queued", message="Queued for semantic ingestion"
            )
            service.schedule(job)
        return {
            "document": service.repository.document(document["id"]),
            "job_id": job["id"] if job else None,
        }

    @app.post("/api/documents/{document_id}/index")
    def reindex(document_id: str):
        document_or_404(document_id)
        job = service.repository.submit(document_id, "ingest")
        service.schedule(job)
        return {"job_id": job["id"]}

    @app.post("/api/documents/{document_id}/approve")
    def approve(document_id: str):
        document = document_or_404(document_id)
        version = document.get("pending_version")
        if not version:
            raise HTTPException(409, "There is no pending document version to review.")
        return service.repository.update_document(
            document_id,
            active_version=version,
            pending_version=None,
            status="ready",
            message="Ready; extraction warnings acknowledged",
        )

    @app.delete("/api/documents/{document_id}")
    def delete(document_id: str):
        document_or_404(document_id)
        try:
            service.repository.delete(document_id)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"deleted": document_id}

    @app.post("/api/chat")
    def answer(request: ChatRequest):
        document = document_or_404(request.document_id)
        if not document.get("active_version"):
            raise HTTPException(
                409, "Index and review the document before asking questions."
            )
        question = request.question.strip()
        if not question:
            raise HTTPException(400, "Enter a question.")
        payload = {
            **request.model_dump(),
            "question": question,
            "version": document["active_version"],
        }
        job = service.repository.submit(document["id"], "answer", payload)
        service.schedule(job)
        return {"job_id": job["id"]}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try:
            return service.repository.job(job_id)
        except KeyError as error:
            raise HTTPException(404, "Job not found.") from error

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8765)
