"""Ingest any supported local document using the semantic graph pipeline."""

import argparse
import json
from pathlib import Path

from legal_rag.pipeline import Ingestion
from legal_rag.retrieval import Models, VectorStore
from legal_rag.settings import Settings
from legal_rag.storage import Repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    repository = Repository(settings)
    document, _ = repository.add(args.path.name, args.path.read_bytes())
    job = repository.submit(document["id"], "ingest")
    vectors = VectorStore(settings.root)
    try:
        result = Ingestion(settings, repository, Models(settings), vectors).run(job)
        repository.update_job(job["id"], status="done", result=result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        repository.update_job(job["id"], status="error", message=str(error))
        current = repository.document(document["id"])
        repository.update_document(
            document["id"],
            status="ready" if current.get("active_version") else "error",
            message=f"Indexing failed: {error}",
        )
        raise
    finally:
        vectors.close()


if __name__ == "__main__":
    main()
