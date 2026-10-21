"""Ask questions against a document's active semantic index."""

import argparse
import json

from legal_rag.agent import research
from legal_rag.retrieval import Models, Retrieval, VectorStore
from legal_rag.settings import Settings
from legal_rag.storage import Repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document_id")
    parser.add_argument("question")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--web", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    repository = Repository(settings)
    document = repository.document(args.document_id)
    models = Models(settings)
    vectors = VectorStore(settings.root)
    try:
        retrieval = Retrieval(
            models, vectors, document, repository.version_folder(document["id"])
        )
        result = (
            retrieval.search(args.question)
            if args.search_only
            else research(args.question, retrieval, settings, print, args.web)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        vectors.close()


if __name__ == "__main__":
    main()
