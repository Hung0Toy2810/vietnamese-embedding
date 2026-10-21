"""Resumable semantic ingestion, orchestrated by LangGraph.

Only artifact paths enter graph state. Large tensors stay on disk between stages;
model-call caches and immutable version folders survive process restarts.
"""

import importlib.metadata
import json
from typing import TypedDict
from uuid import uuid4

import networkx as nx
import torch
from langgraph.graph import END, START, StateGraph

from legal_rag.common import save_json
from legal_rag.parsing import parse_document, preview_html
from legal_rag.llm import generation_options, model_identity
from legal_rag.retrieval import Models, VectorStore
from legal_rag.schemas import Atom, ChunkPlan, ParsedDocument
from legal_rag.semantic import (
    atomize,
    build_graph,
    candidate_pairs,
    identity,
    make_chunks,
    plan_chunks,
)
from legal_rag.settings import Settings
from legal_rag.storage import Repository
from utils import atomic_save


class State(TypedDict):
    job_id: str
    document_id: str
    version: str


class Ingestion:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        models: Models,
        vectors: VectorStore,
    ):
        self.settings, self.repository, self.models, self.vectors = (
            settings,
            repository,
            models,
            vectors,
        )
        workflow = StateGraph(State)
        steps = ["parse", "encode_units", "plan", "relate", "chunk", "index", "publish"]
        for name in steps:
            workflow.add_node(name, getattr(self, name))
        workflow.add_edge(START, steps[0])
        for left, right in zip(steps, steps[1:]):
            workflow.add_edge(left, right)
        workflow.add_edge(steps[-1], END)
        self.workflow = workflow.compile()

    def folder(self, state: State):
        return self.repository.version_folder(state["document_id"], state["version"])

    def read(self, state: State, name: str):
        return json.loads((self.folder(state) / name).read_text())

    def progress(self, state: State, message: str):
        self.repository.update_job(state["job_id"], status="running", message=message)
        self.repository.update_document(state["document_id"], message=message)

    def run(self, job: dict) -> dict:
        self.settings = self.settings.model_copy(
            update={"model_revision": model_identity(self.settings)}
        )
        version = job["payload"].get("work_version") or uuid4().hex
        payload = {**job["payload"], "work_version": version}
        self.repository.update_job(job["id"], payload=payload)
        folder = self.repository.version_folder(job["document_id"], version)
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": 3,
            "semantic_policy": "language_windows_and_distant_pairs_v1",
            "generation_options": generation_options(self.settings),
            "settings": self.settings.model_dump(mode="json"),
            "models": self.models.fingerprint,
            "parser_version": importlib.metadata.version("docling")
            if self.settings.parser == "docling"
            else "plain-v1",
        }
        manifest_path = folder / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Ingestion configuration changed; start a new indexing job."
            )
        save_json(manifest_path, manifest)
        state: State = {
            "job_id": job["id"],
            "document_id": job["document_id"],
            "version": version,
        }
        self.repository.update_document(
            job["document_id"],
            status="processing",
            message="Starting semantic ingestion",
        )
        self.workflow.invoke(state)
        return self.repository.document(job["document_id"])

    def parse(self, state: State):
        self.progress(state, "Extracting document text and source locations")
        folder = self.folder(state)
        if not (folder / "parsed.json").exists():
            parsed = parse_document(
                self.repository.source(state["document_id"]), self.settings
            )
            save_json(folder / "parsed.json", parsed.model_dump())
            name = self.repository.document(state["document_id"])["name"]
            (folder / "preview.html").write_text(preview_html(parsed, name))
        return {}

    def encode_units(self, state: State):
        self.progress(state, "Encoding language units without layout-based grouping")
        folder = self.folder(state)
        if not (folder / "atoms.json").exists():
            parsed = ParsedDocument.model_validate(self.read(state, "parsed.json"))
            atoms = atomize(
                parsed,
                state["document_id"],
                self.models.tokenizer,
                self.settings.atom_tokens,
            )
            save_json(folder / "atoms.json", [atom.model_dump() for atom in atoms])
        if not (folder / "units.pt").exists():
            atoms = [Atom.model_validate(row) for row in self.read(state, "atoms.json")]
            vectors = self.models.encode([atom.text for atom in atoms])
            atomic_save({"vectors": vectors}, folder / "units.pt")

    def plan(self, state: State):
        self.progress(state, "Planning chunks from semantic diversity and dependencies")
        folder = self.folder(state)
        if (folder / "plan.json").exists():
            return {}
        atoms = [Atom.model_validate(row) for row in self.read(state, "atoms.json")]
        vectors = torch.load(folder / "units.pt", weights_only=True)["vectors"]
        pairs = candidate_pairs(vectors, self.settings.candidate_neighbors)
        plan = plan_chunks(atoms, vectors, pairs, self.settings)
        save_json(folder / "pairs.json", pairs)
        save_json(folder / "plan.json", plan.model_dump())
        return {}

    def relate(self, state: State):
        folder = self.folder(state)
        if (folder / "graph.json").exists():
            return {}
        atoms = [Atom.model_validate(row) for row in self.read(state, "atoms.json")]
        plan = ChunkPlan.model_validate(self.read(state, "plan.json"))
        cache = lambda key, compute: self.repository.cache(
            state["document_id"], key, compute
        )
        graph = build_graph(
            atoms,
            self.read(state, "pairs.json"),
            plan,
            self.settings,
            lambda message: self.progress(state, message),
            cache,
        )
        graph.graph.update(
            document_id=state["document_id"],
            version=state["version"],
            plan_id=identity(json.dumps(plan.model_dump(), sort_keys=True)),
            note="LLM relations are hypotheses; endpoint evidence is copied from original units, not generated or proven entailment.",
        )
        save_json(folder / "graph.json", nx.node_link_data(graph, edges="edges"))
        return {}

    def chunk(self, state: State):
        self.progress(
            state, "Creating source-backed semantic parents and encoder-sized children"
        )
        folder = self.folder(state)
        if (folder / "index.json").exists():
            return {}
        atoms = [Atom.model_validate(row) for row in self.read(state, "atoms.json")]
        graph = nx.node_link_graph(self.read(state, "graph.json"), edges="edges")
        plan = ChunkPlan.model_validate(self.read(state, "plan.json"))
        document = self.repository.document(state["document_id"])
        parents, children = make_chunks(
            graph, atoms, plan, self.models.tokenizer, document["id"], document["name"]
        )
        save_json(
            folder / "index.json",
            {
                "schema": 2,
                "document_id": document["id"],
                "version": state["version"],
                "model_fingerprint": self.models.fingerprint,
                "parents": parents,
                "children": children,
            },
        )
        return {}

    def index(self, state: State):
        self.progress(
            state, "Indexing semantic children in the version-scoped vector store"
        )
        folder = self.folder(state)
        if (folder / "indexed.json").exists():
            return {}
        index = self.read(state, "index.json")
        children = index["children"]
        for start in range(0, len(children), 32):
            batch = children[start : start + 32]
            vectors = self.models.encode([child["text"] for child in batch])
            self.vectors.put(state["document_id"], state["version"], batch, vectors)
        save_json(folder / "indexed.json", {"children": len(children)})
        return {}

    def publish(self, state: State):
        parsed = ParsedDocument.model_validate(self.read(state, "parsed.json"))
        warnings = parsed.warnings
        values = {
            "pages": parsed.pages,
            "text_pages": len({span.page for span in parsed.spans if span.page}),
            "warnings": warnings,
            "pending_version": state["version"] if warnings else None,
            "status": "review" if warnings else "ready",
            "message": "Review extraction warnings before activation"
            if warnings
            else "Ready for questions",
        }
        if not warnings:
            values["active_version"] = state["version"]
        self.repository.update_document(state["document_id"], **values)
        return {}
