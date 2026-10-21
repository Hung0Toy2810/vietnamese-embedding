"""Small, manually specified development checks; not a held-out benchmark.

Run from the repository root:
python reports/graph-quality-20261022/evaluate.py WORKSPACE OUTPUT [--answers]
Gold atom positions refer only to the frozen ten-page extraction in this folder.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from legal_rag.agent import research
from legal_rag.grounding import ANSWER_PROMPT
from legal_rag.llm import generation_options
from legal_rag.retrieval import Models, Retrieval, VectorStore
from legal_rag.settings import Settings
from legal_rag.storage import Repository


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--answers", action="store_true")
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(2)
    settings = Settings(root=args.workspace, answer_thinking=args.thinking)
    repo = Repository(settings)
    document = next(doc for doc in repo.documents() if doc.get("active_version"))
    folder = repo.version_folder(document["id"])
    atoms = json.loads((folder / "atoms.json").read_text())
    gold = json.loads(Path(__file__).with_name("baseline-atoms.json").read_text())
    assert [a["text"] for a in atoms] == [a["text"] for a in gold]
    store = VectorStore(settings.root)
    try:
        retrieval = Retrieval(Models(settings), store, document, folder)
        semantic_edges = [
            (a, b, d)
            for a, b, d in retrieval.graph.edges(data=True)
            if d.get("semantic", d.get("grouping"))
        ]
        connected = {node for a, b, _ in semantic_edges for node in (a, b)}
        results = {
            "document": document,
            "generation": {
                "model": settings.model,
                "reasoning": settings.answer_thinking,
                "options": generation_options(settings, answer=True),
                "answer_prompt_sha256": hashlib.sha256(
                    ANSWER_PROMPT.encode()
                ).hexdigest(),
            },
            "graph": {
                "nodes": len(atoms),
                "semantic_edges": len(semantic_edges),
                "grouping_edges": sum(
                    bool(d.get("grouping")) for _, _, d in semantic_edges
                ),
                "navigation_edges": sum(
                    not d.get("grouping") for _, _, d in semantic_edges
                ),
                "isolated_nodes": len(atoms) - len(connected),
            },
            "cases": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for case in json.loads(Path(__file__).with_name("cases.json").read_text()):
            if args.cases and case["id"] not in args.cases:
                continue
            start = time.monotonic()
            children = retrieval.search(case["question"])
            parents = retrieval.expand(children)
            expected = {atoms[i]["id"] for i in case["gold_atoms"]}
            found = {node for parent in parents for node in parent["atom_ids"]}
            row = {
                **case,
                "reranked_atoms": [
                    node for child in children for node in child["atom_ids"]
                ],
                "context_atoms": sorted(found),
                "gold_context_coverage": len(expected & found) / len(expected)
                if expected
                else None,
            }
            if args.answers:
                try:
                    row["result"] = research(
                        case["question"],
                        retrieval,
                        settings,
                        report=lambda text: print(text, flush=True),
                    )
                    read = {
                        node
                        for parent in row["result"]["parents"]
                        for node in parent["atom_ids"]
                    }
                    row["gold_read_coverage"] = (
                        len(expected & read) / len(expected) if expected else None
                    )
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    if error.__cause__ is not None:
                        row["cause"] = str(error.__cause__)
            row["seconds"] = round(time.monotonic() - start, 2)
            results["cases"].append(row)
            args.output.write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n"
            )
            print(case["id"], row["gold_context_coverage"], row["seconds"], flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
