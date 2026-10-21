"""Compare graph revisions with the same extraction, embeddings and chunk plan.

This controlled development experiment reuses expensive, unchanged input stages.
Production ingestion still runs the complete pipeline for newly uploaded files.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from legal_rag.pipeline import Ingestion
from legal_rag.retrieval import Models, VectorStore
from legal_rag.settings import Settings
from legal_rag.storage import Repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    previous = Repository(Settings(root=args.baseline))
    original = previous.documents()[0]
    if original.get("active_version"):
        baseline = previous.version_folder(original["id"])
    else:
        baseline = next(
            (previous.folder(original["id"]) / "versions").glob("*/plan.json")
        ).parent
    settings = Settings(root=args.destination)
    repo = Repository(settings)
    document, _ = repo.add(
        original["name"], previous.source(original["id"]).read_bytes()
    )
    job = repo.submit(document["id"], "ingest")
    version = job["payload"].get("work_version") or uuid4().hex
    job = repo.update_job(job["id"], payload={"work_version": version})
    folder = repo.version_folder(document["id"], version)
    folder.mkdir(parents=True, exist_ok=True)
    for name in (
        "parsed.json",
        "preview.html",
        "atoms.json",
        "units.pt",
        "pairs.json",
        "plan.json",
    ):
        if not (folder / name).exists():
            shutil.copyfile(baseline / name, folder / name)
    # The new subject groups do not use the old community-resolution parameter.
    plan_path = folder / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan.pop("community_resolution", None)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    store = VectorStore(settings.root)
    try:
        pipeline = Ingestion(settings, repo, Models(settings), store)
        result = pipeline.run(job)
        repo.update_job(job["id"], status="done", result=result)
        print(result, flush=True)
    except Exception as error:
        repo.update_job(job["id"], status="error", message=str(error))
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
