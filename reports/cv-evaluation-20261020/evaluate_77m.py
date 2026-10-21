"""Reproducible legal-retrieval evaluation; never writes to checkpoints/."""

import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from checkpoint import load_model
from evaluate import ranking_metrics
from grad_cache import encode
from utils import DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER, file_hash

REPO = "GreenNode/zalo-ai-legal-text-retrieval-vn"
REVISION = "12d76d4d04b94ceada970fcfbe7fec20bfa97389"
OUTPUT = Path(__file__).resolve().parent


def split_child(text, tokenizer, limit=224, overlap=40):
    """Frozen October 20 evaluation windows; independent of application chunking."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(encoded.ids):
        end = min(start + limit, len(encoded.ids))
        left = encoded.offsets[start][0]
        right = encoded.offsets[end - 1][1]
        chunks.append(text[left:right])
        if end == len(encoded.ids):
            break
        start = end - overlap
    return chunks


def main():
    started = time.time()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    checkpoint_hash = file_hash(DEFAULT_CHECKPOINT)
    tables, hashes = {}, {}
    for name in ("corpus", "queries", "qrels"):
        path = hf_hub_download(
            REPO,
            f"{name}/test-00000-of-00001.parquet",
            repo_type="dataset",
            revision=REVISION,
        )
        tables[name] = pq.read_table(path).to_pylist()
        hashes[name] = file_hash(Path(path))

    relevant = {}
    for row in tables["qrels"]:
        if row["score"] > 0:
            relevant.setdefault(row["query-id"], set()).add(row["corpus-id"])
    positives = set().union(*relevant.values())
    corpus = {row["id"]: row for row in tables["corpus"]}
    assert positives <= corpus.keys()
    # Fixed before scoring: all labelled positives, then hash-selected distractors.
    distractors = sorted(
        corpus.keys() - positives,
        key=lambda key: hashlib.sha256(("42:" + key).encode()).hexdigest(),
    )
    document_ids = sorted(positives | set(distractors[: 5000 - len(positives)]))
    query_lookup = {}
    for query in tables["queries"]:
        if query["id"] in query_lookup:
            assert query_lookup[query["id"]]["text"] == query["text"]
        query_lookup[query["id"]] = query
    assert relevant.keys() <= query_lookup.keys()
    queries = [query_lookup[key] for key in sorted(relevant)]
    protocol = {
        "dataset": REPO,
        "revision": REVISION,
        "split": "test",
        "file_sha256": hashes,
        "full_corpus_count": len(corpus),
        "candidate_count": len(document_ids),
        "query_count": len(queries),
        "raw_query_rows": len(tables["queries"]),
        "duplicate_query_id_rows": len(tables["queries"]) - len(query_lookup),
        "positive_document_count": len(positives),
        "seed": 42,
        "selection": "All labelled positives plus SHA256('42:' + id) sorted distractors.",
        "encoding": "title + text; 224-token child windows, overlap 40; no truncation",
        "scoring": "maximum child cosine per document; document-ID order breaks ties",
        "checkpoint_sha256": checkpoint_hash,
        "tokenizer_sha256": file_hash(DEFAULT_TOKENIZER),
        "document_ids": document_ids,
        "query_ids": [q["id"] for q in queries],
        "limitations": [
            "5,000-document subset, not the full MTEB/Zalo corpus benchmark.",
            "No complete historical training text is available to prove absence of overlap.",
            "Document-level max-over-chunks differs from single-vector MTEB encoding.",
            "Encoder only; no reranker or LLM is used in these metrics.",
        ],
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2)
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, saved = load_model(DEFAULT_CHECKPOINT, device)
    model.eval().requires_grad_(False)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records, owners = [], []
    for number, document_id in enumerate(document_ids):
        row = corpus[document_id]
        for text in split_child(row["title"] + "\n" + row["text"], tokenizer):
            ids = tokenizer.encode(text).ids
            assert 0 < len(ids) <= model.config.max_len
            records.append({"ids": ids})
            owners.append(number)
    query_records = [{"ids": tokenizer.encode(q["text"]).ids} for q in queries]
    assert all(0 < len(q["ids"]) <= model.config.max_len for q in query_records)
    print(
        f"Device={device}, queries={len(queries)}, documents={len(document_ids)}, children={len(records)}",
        flush=True,
    )

    def vectors(items):
        # Sort by length for less padding, then restore the original order.
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
        result = torch.empty(len(items), model.config.output_size)
        with torch.inference_mode():
            for start in range(0, len(order), 32):
                batch = order[start : start + 32]
                result[batch] = encode(
                    model, [items[i] for i in batch], device, 32, bf16=False
                ).cpu()
                if start % 512 == 0:
                    print(
                        f"Encoded {min(start + 32, len(items))}/{len(items)}; elapsed {time.time() - started:.1f}s",
                        flush=True,
                    )
        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            result.norm(dim=1), torch.ones(len(items)), atol=1e-5, rtol=1e-5
        )
        return result

    documents = vectors(records)
    query_vectors = vectors(query_records)
    ownership = torch.tensor(owners)
    rankings = []
    for start in range(0, len(queries), 32):
        child_scores = query_vectors[start : start + 32] @ documents.T
        scores = torch.full((len(child_scores), len(document_ids)), -torch.inf)
        scores.scatter_reduce_(
            1,
            ownership.expand(len(scores), -1),
            child_scores,
            reduce="amax",
            include_self=True,
        )
        rankings.extend(
            scores.argsort(dim=1, descending=True, stable=True)[:, :100].tolist()
        )
    lookup = {name: i for i, name in enumerate(document_ids)}
    truths = [{lookup[p] for p in relevant[q["id"]]} for q in queries]
    metrics = ranking_metrics(rankings, truths)
    metrics["recall100"] = sum(
        len(set(r[:100]) & t) / len(t) for r, t in zip(rankings, truths)
    ) / len(truths)
    metrics["hit100"] = sum(
        bool(set(r[:100]) & t) for r, t in zip(rankings, truths)
    ) / len(truths)
    result = {
        "checkpoint_step": saved["step"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "query_count": len(queries),
        "candidate_count": len(document_ids),
        "child_count": len(records),
        "metrics": metrics,
        "seconds": time.time() - started,
    }
    assert file_hash(DEFAULT_CHECKPOINT) == checkpoint_hash
    (OUTPUT / "encoder-result.json").write_text(json.dumps(result, indent=2))
    torch.save(
        {
            "queries": query_vectors,
            "children": documents,
            "owners": ownership,
            "rankings": rankings,
            "truths": [sorted(t) for t in truths],
        },
        OUTPUT / "evaluation-tensors.pt",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
