"""Small shared helpers for durable files and content identities."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints/best-step-190399.pt"
DEFAULT_TOKENIZER = ROOT / "data/production-tokenizer.json"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def atomic_save(data: dict, path: Path) -> None:
    import torch

    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(data, stream)
        stream.flush()
        import os

        os.fsync(stream.fileno())
    temporary.replace(path)
