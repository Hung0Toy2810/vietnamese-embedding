"""HTTP, durable jobs, review gates and document isolation without live models."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from legal_rag.settings import Settings
from legal_rag.storage import Repository
from legal_rag.webapp import create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = Settings(root=Path(self.temporary.name), parser="plain")
        self.app = create_app(settings=self.settings)
        self.scheduling = patch.object(self.app.state.service, "schedule")
        self.scheduling.start()
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.scheduling.stop()
        self.temporary.cleanup()

    def upload(self, name="example.txt", text="Example document text."):
        return self.client.post(
            "/api/documents", files={"file": (name, text.encode(), "text/plain")}
        )

    def test_empty_catalog_does_not_require_a_law_index_or_models(self):
        self.assertEqual(self.client.get("/api/documents").json(), [])
        self.assertIsNone(self.app.state.service.models)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_upload_is_content_deduplicated_and_jobs_persist(self):
        first = self.upload().json()
        second = self.upload().json()
        self.assertEqual(first["document"]["id"], second["document"]["id"])
        repo = Repository(self.settings)
        self.assertEqual(len(repo.documents()), 1)
        self.assertEqual(repo.job(first["job_id"])["status"], "queued")
        self.assertEqual(len(repo.jobs()), 1)

    def test_unsupported_fake_and_empty_files_are_rejected(self):
        self.assertEqual(self.upload("bad.exe").status_code, 400)
        self.assertEqual(self.upload("bad.pdf").status_code, 400)
        self.assertEqual(self.upload(text="").status_code, 400)

    def test_unindexed_document_cannot_be_queried(self):
        document = self.upload().json()["document"]
        self.assertEqual(
            self.client.post(
                "/api/chat",
                json={"document_id": document["id"], "question": "Example?"},
            ).status_code,
            409,
        )

    def test_answers_pin_the_active_version_and_validate_history(self):
        document = self.upload().json()["document"]
        repo = self.app.state.service.repository
        repo.update_document(document["id"], active_version="a" * 32, status="ready")
        response = self.client.post(
            "/api/chat", json={"document_id": document["id"], "question": "Example?"}
        )
        job = repo.job(response.json()["job_id"])
        self.assertEqual(job["payload"]["version"], "a" * 32)
        self.assertEqual(
            self.client.post(
                "/api/chat", json={"document_id": document["id"], "question": " "}
            ).status_code,
            400,
        )
        self.assertEqual(
            self.client.post(
                "/api/chat",
                json={
                    "document_id": document["id"],
                    "question": "Example?",
                    "history": [{"role": "user", "content": "x"}] * 5,
                },
            ).status_code,
            422,
        )

    def test_preview_is_inert_and_unknown_versions_are_rejected(self):
        document = self.upload().json()["document"]
        repo = self.app.state.service.repository
        self.assertEqual(
            self.client.get("/api/preview/" + document["id"]).status_code, 409
        )
        folder = repo.version_folder(document["id"], "b" * 32)
        folder.mkdir(parents=True)
        (folder / "preview.html").write_text("<pre>Example</pre>")
        repo.update_document(document["id"], pending_version="b" * 32, status="review")
        response = self.client.get("/api/preview/" + document["id"])
        self.assertEqual(response.status_code, 200)
        self.assertIn("default-src 'none'", response.headers["content-security-policy"])
        self.assertEqual(
            self.client.get(
                "/api/preview/" + document["id"] + "?version=../../secret"
            ).status_code,
            409,
        )
        self.assertEqual(
            self.client.post("/api/documents/" + document["id"] + "/approve").json()[
                "active_version"
            ],
            "b" * 32,
        )

    def test_unknown_documents_and_jobs_return_404(self):
        self.assertEqual(self.client.get("/api/jobs/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/files/missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()
