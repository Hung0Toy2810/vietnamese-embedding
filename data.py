from functools import lru_cache
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch
from torch import Tensor

from config import Config


class Corpus:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["format"] != 2:
            raise ValueError("Unsupported prepared dataset format")
        self.config = Config.from_dict(self.metadata["config"])
        self.db = sqlite3.connect(
            (root / "dataset.sqlite").resolve().as_uri() + "?mode=ro", uri=True
        )

    def close(self) -> None:
        self.record.cache_clear()
        self.db.close()

    def row_ids(self, split: str) -> list[int]:
        return [
            row[0]
            for row in self.db.execute("SELECT id FROM pairs WHERE split=? ORDER BY id", (split,))
        ]

    @lru_cache(maxsize=4096)
    def record(self, text_id: int) -> dict:
        row = self.db.execute("SELECT tokens FROM texts WHERE id=?", (text_id,)).fetchone()
        if row is None:
            raise ValueError(f"Missing text {text_id}")
        ids = row[0]
        return {
            "id": text_id,
            "ids": np.frombuffer(ids, dtype=np.int32).astype(np.int64),
        }

    def relevant(self, query: int) -> set[int]:
        return {
            row[0]
            for row in self.db.execute("SELECT document FROM positives WHERE query=?", (query,))
        }

    def batch(self, row_ids: list[int], device: torch.device):
        pairs = [
            self.db.execute(
                "SELECT query,positive,negatives FROM pairs WHERE id=?", (i,)
            ).fetchone()
            for i in row_ids
        ]
        documents = list(
            dict.fromkeys(
                i for _, positive, negatives in pairs for i in [positive, *json.loads(negatives)]
            )
        )
        queries = [query for query, _, _ in pairs]
        known = [self.relevant(query) for query in queries]
        positives = torch.tensor(
            [[doc in relevant for doc in documents] for relevant in known],
            dtype=torch.bool,
            device=device,
        )
        records = [self.record(i) for i in queries + documents]
        return records, len(queries), positives


def padded(records: list[dict], c: Config, device: torch.device) -> tuple[Tensor, Tensor]:
    length = max(len(row["ids"]) for row in records)
    if length > c.max_len:
        raise ValueError("Record exceeds model context length")
    ids = np.full((len(records), length), c.pad_id, dtype=np.int64)
    for i, row in enumerate(records):
        ids[i, : len(row["ids"])] = row["ids"]
    tensor = torch.from_numpy(ids).to(device)
    return tensor, tensor.ne(c.pad_id)
