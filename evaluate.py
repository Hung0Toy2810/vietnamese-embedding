"""Evaluate the prepared validation split against its complete candidate corpus."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from checkpoint import check_compatible, load_model
from data import Corpus
from grad_cache import encode
from utils import DEFAULT_CHECKPOINT, file_hash, write_json


def ranking_metrics(top_ids: list[list[int]], truths: list[set[int]]) -> dict:
    if not truths or len(top_ids) != len(truths) or any(not truth for truth in truths):
        raise ValueError("Every query needs relevant candidates")
    metrics = {
        "mrr10": [],
        "ndcg10": [],
        **{f"{metric}{k}": [] for metric in ("hit", "recall") for k in (1, 5, 10)},
    }
    for ranking, truth in zip(top_ids, truths):
        hits = [int(i in truth) for i in ranking[:10]]
        metrics["mrr10"].append(next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0))
        dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(truth))))
        metrics["ndcg10"].append(dcg / ideal)
        for k in (1, 5, 10):
            found = len(set(ranking[:k]) & truth)
            metrics[f"hit{k}"].append(float(found > 0))
            metrics[f"recall{k}"].append(found / len(truth))
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def search(queries: torch.Tensor, documents: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    scores = torch.empty((len(queries), 0), device=queries.device)
    indices = torch.empty((len(queries), 0), dtype=torch.long, device=queries.device)
    for start in range(0, len(documents), chunk_size):
        block = documents[start : start + chunk_size].to(queries.device)
        values = torch.cat((scores, queries @ block.T), dim=1)
        ids = torch.arange(start, start + len(block), device=queries.device).expand(
            len(queries), -1
        )
        ids = torch.cat((indices, ids), dim=1)
        # Earlier document IDs win exact ties; the corpus is sorted by ID.
        chosen = values.argsort(dim=1, descending=True, stable=True)[:, :10]
        scores, indices = values.gather(1, chosen), ids.gather(1, chosen)
    return indices.cpu()


def evaluate(data: Path, checkpoint: Path, device: str = "cpu", batch_size: int = 32) -> dict:
    corpus = Corpus(data)
    try:
        model, saved = load_model(checkpoint, torch.device(device))
        check_compatible(model.config, corpus.metadata, data / "tokenizer.json", saved)
        if file_hash(data / "dataset.sqlite") != corpus.metadata["database_sha256"]:
            raise ValueError("Prepared data has changed")
        model.eval().requires_grad_(False)
        pairs = list(
            corpus.db.execute("SELECT query,positive,negatives FROM pairs WHERE split='validation'")
        )
        query_ids = sorted({q for q, _, _ in pairs})
        document_ids = sorted({i for _, p, negatives in pairs for i in [p, *json.loads(negatives)]})
        if not query_ids or not document_ids:
            raise ValueError(
                "Validation split is empty; prepare more independent query/passage groups"
            )
        with torch.inference_mode():

            def vectors(ids):
                return torch.cat(
                    [
                        encode(
                            model,
                            [corpus.record(i) for i in ids[start : start + batch_size]],
                            torch.device(device),
                            batch_size,
                            bf16=False,
                        ).cpu()
                        for start in range(0, len(ids), batch_size)
                    ]
                )

            documents = vectors(document_ids)
            rankings = []
            for start in range(0, len(query_ids), batch_size):
                queries = vectors(query_ids[start : start + batch_size]).to(device)
                rankings.extend(
                    [[document_ids[i] for i in row] for row in search(queries, documents).tolist()]
                )
        truths = [corpus.relevant(q) & set(document_ids) for q in query_ids]
        return {
            "split": "validation",
            "checkpoint_step": saved["step"],
            "query_count": len(query_ids),
            "candidate_count": len(document_ids),
            "dataset_id": corpus.metadata["dataset_id"],
            "metrics": ranking_metrics(rankings, truths),
        }
    finally:
        corpus.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/retrieval"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Refusing to overwrite an existing report")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    result = evaluate(args.data, args.checkpoint, args.device, args.batch_size)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
