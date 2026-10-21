"""SQLite metadata and durable jobs; immutable per-version source artifacts."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4
from typing import Literal

from legal_rag.common import save_json
from legal_rag.parsing import SUPPORTED
from legal_rag.schemas import Job
from legal_rag.settings import Settings


class Repository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.root
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "catalog.sqlite"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            """)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=30)

    def folder(self, document_id: str) -> Path:
        if len(document_id) != 24 or any(
            c not in "0123456789abcdef" for c in document_id
        ):
            raise KeyError("Invalid document identifier.")
        return self.root / "documents" / document_id

    def documents(self) -> list[dict]:
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT data FROM documents ORDER BY rowid")
            ]

    def document(self, document_id: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT data FROM documents WHERE id=?", (document_id,)
            ).fetchone()
        if row is None:
            raise KeyError("Document not found.")
        return json.loads(row[0])

    def update_document(self, document_id: str, **values) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT data FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Document not found.")
            document = {**json.loads(row[0]), **values}
            db.execute(
                "UPDATE documents SET data=? WHERE id=?",
                (json.dumps(document, ensure_ascii=False), document_id),
            )
        return document

    def add(self, name: str, content: bytes) -> tuple[dict, bool]:
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError(f"Supported formats: {', '.join(sorted(SUPPORTED))}")
        if not content or len(content) > self.settings.max_upload_bytes:
            raise ValueError(
                "The upload is empty or exceeds the configured size limit."
            )
        if suffix == ".pdf" and not content.startswith(b"%PDF"):
            raise ValueError("The uploaded file is not a PDF.")
        digest = hashlib.sha256(content).hexdigest()
        document_id = digest[:24]
        folder = self.folder(document_id)
        folder.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT data FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            if row:
                document = json.loads(row[0])
                return document, not document.get("active_version") and document[
                    "status"
                ] not in {"queued", "processing"}
            path = folder / ("source" + suffix)
            path.write_bytes(content)
            document = {
                "id": document_id,
                "name": Path(name).name[:180],
                "suffix": suffix,
                "sha256": digest,
                "size": len(content),
                "pages": 0,
                "text_pages": 0,
                "status": "unindexed",
                "message": "Ready to index",
                "active_version": None,
                "version": digest[:12],
                "warnings": [],
                "builtin": False,
                "created": time.time(),
                "access_scope": "local",
            }
            db.execute(
                "INSERT INTO documents VALUES (?, ?)",
                (document_id, json.dumps(document, ensure_ascii=False)),
            )
        return document, True

    def source(self, document_id: str) -> Path:
        document = self.document(document_id)
        return self.folder(document_id) / ("source" + document["suffix"])

    def version_folder(self, document_id: str, version: str | None = None) -> Path:
        version = version or self.document(document_id).get("active_version")
        if (
            not version
            or len(version) != 32
            or any(c not in "0123456789abcdef" for c in version)
        ):
            raise ValueError("The document has no active semantic index.")
        return self.folder(document_id) / "versions" / version

    def cache(self, document_id: str, key: str, compute) -> dict:
        path = self.folder(document_id) / "model-cache" / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())
        result = compute()
        save_json(path, result)
        return result

    def submit(
        self,
        document_id: str,
        kind: Literal["ingest", "answer"],
        payload: dict | None = None,
    ) -> dict:
        self.document(document_id)
        job = Job(
            id=uuid4().hex,
            document_id=document_id,
            kind=kind,
            payload=payload or {},
            created=time.time(),
        )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if kind == "ingest":
                for row in db.execute("SELECT data FROM jobs"):
                    existing = json.loads(row[0])
                    if (
                        existing["document_id"] == document_id
                        and existing["kind"] == kind
                        and existing["status"] in {"queued", "running"}
                    ):
                        return existing
            db.execute(
                "INSERT INTO jobs VALUES (?, ?)", (job.id, job.model_dump_json())
            )
        return job.model_dump()

    def jobs(self) -> list[dict]:
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT data FROM jobs ORDER BY rowid")
            ]

    def job(self, job_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("Job not found.")
        return json.loads(row[0])

    def update_job(self, job_id: str, **values) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("Job not found.")
            job = {**json.loads(row[0]), **values}
            db.execute(
                "UPDATE jobs SET data=? WHERE id=?",
                (json.dumps(job, ensure_ascii=False), job_id),
            )
        return job

    def delete(self, document_id: str) -> None:
        """Hide the catalog entry; version artifacts are retained for audit/recovery."""
        with self.connect() as db:
            for job in self.jobs():
                if job["document_id"] == document_id and job["status"] in {
                    "queued",
                    "running",
                }:
                    raise ValueError(
                        "Wait for the document's active jobs to finish before deleting it."
                    )
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))
