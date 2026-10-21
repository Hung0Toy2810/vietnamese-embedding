"""Real storage/vector/workflow integration with isolated deterministic models."""

import tempfile
import unittest
from typing import cast
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import torch
from tokenizers import Tokenizer

from legal_rag.pipeline import Ingestion
from legal_rag.retrieval import Models, Retrieval, VectorStore
from legal_rag.schemas import ChunkPlan, ParsedDocument, Span
from legal_rag.settings import Settings
from legal_rag.storage import Repository
from utils import DEFAULT_TOKENIZER


class FakeModels:
    def __init__(self, settings):
        self.settings = settings
        self.tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
        self.fingerprint = {
            "checkpoint_sha256": "fixture",
            "tokenizer_sha256": "fixture",
        }
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        vectors = torch.zeros(len(texts), 768)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                vectors[i, sum(word.encode()) % 768] += 1
        return torch.nn.functional.normalize(vectors, dim=1)

    def rerank(self, question, candidates):
        return candidates[: self.settings.rerank_limit]


def empty_semantic_graph(atoms, *args):
    graph = nx.MultiDiGraph()
    for atom in atoms:
        graph.add_node(atom.id, kind="text_unit", **atom.model_dump(exclude={"id"}))
    return graph


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            root=Path(self.temp.name), parser="plain", model_revision="fixture"
        )
        self.repo = Repository(self.settings)
        self.store = VectorStore(self.settings.root)
        self.models = cast(Models, FakeModels(self.settings))
        self.plan = patch(
            "legal_rag.pipeline.plan_chunks",
            return_value=ChunkPlan(rationale="Fixture plan"),
        )
        self.relations = patch(
            "legal_rag.pipeline.build_graph", side_effect=empty_semantic_graph
        )
        self.plan.start()
        self.relations.start()

    def tearDown(self):
        self.plan.stop()
        self.relations.stop()
        self.store.close()
        self.temp.cleanup()

    def ingest(self, text):
        document, _ = self.repo.add("document.txt", text.encode())
        job = self.repo.submit(document["id"], "ingest")
        result = Ingestion(self.settings, self.repo, self.models, self.store).run(job)
        self.repo.update_job(job["id"], status="done", result=result)
        return result, job

    def test_workflow_indexes_without_builtin_law_and_scopes_versions(self):
        first, _ = self.ingest("Customers may return products within seven days.")
        second, _ = self.ingest("Disconnect power before cleaning the machine.")
        retrieval = Retrieval(
            self.models, self.store, first, self.repo.version_folder(first["id"])
        )
        candidates = retrieval.search("return products")
        self.assertTrue(candidates)
        self.assertTrue(all(item["document_id"] == first["id"] for item in candidates))
        old = first["active_version"]
        version = "f" * 32
        self.store.put(
            first["id"], version, [{"id": "unpublished"}], torch.ones(1, 768)
        )
        keys = self.store.search(first["id"], old, torch.ones(768), 100)
        self.assertNotIn("unpublished", keys)
        self.assertNotEqual(first["id"], second["id"])

    def test_incomplete_ingestion_preserves_previous_active_version(self):
        document, _ = self.ingest("Returns are accepted within seven days.")
        job = self.repo.submit(document["id"], "ingest")
        pipeline = Ingestion(self.settings, self.repo, self.models, self.store)
        with patch.object(
            self.store, "put", side_effect=RuntimeError("disk unavailable")
        ):
            with self.assertRaisesRegex(RuntimeError, "disk unavailable"):
                pipeline.run(job)
        self.assertEqual(
            self.repo.document(document["id"])["active_version"],
            document["active_version"],
        )
        updated = self.repo.job(job["id"])
        parsed = (
            self.repo.version_folder(document["id"], updated["payload"]["work_version"])
            / "parsed.json"
        )
        before = parsed.stat().st_mtime_ns
        result = pipeline.run(updated)
        self.assertEqual(parsed.stat().st_mtime_ns, before)
        self.assertNotEqual(result["active_version"], document["active_version"])

    def test_extraction_warnings_require_review_before_activation(self):
        document, _ = self.repo.add("warning.txt", b"Example source content.")
        job = self.repo.submit(document["id"], "ingest")
        parsed = ParsedDocument(
            text="Example source content.",
            spans=[Span(start=0, end=23)],
            parser="fixture",
            warnings=["Page 2 could not be read"],
        )
        with patch("legal_rag.pipeline.parse_document", return_value=parsed):
            result = Ingestion(self.settings, self.repo, self.models, self.store).run(
                job
            )
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["active_version"])
        self.assertTrue(result["pending_version"])

    def test_changed_model_fingerprint_requires_reindexing(self):
        document, _ = self.ingest("A product returns policy.")
        self.models.fingerprint = {
            "checkpoint_sha256": "different",
            "tokenizer_sha256": "fixture",
        }
        with self.assertRaisesRegex(ValueError, "Reindex"):
            Retrieval(
                self.models,
                self.store,
                document,
                self.repo.version_folder(document["id"]),
            )

    def test_duplicate_ingestion_jobs_are_not_scheduled_twice(self):
        document, _ = self.repo.add("file.txt", b"Example source.")
        first = self.repo.submit(document["id"], "ingest")
        second = self.repo.submit(document["id"], "ingest")
        self.assertEqual(first["id"], second["id"])


if __name__ == "__main__":
    unittest.main()
