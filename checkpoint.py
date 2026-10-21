"""Load legacy weights and save new runs without modifying the packaged checkpoint."""

from dataclasses import asdict
from pathlib import Path

import torch

from config import Config
from model import Student
from utils import DEFAULT_TOKENIZER, atomic_save, file_hash


def load_model(path: Path, device: torch.device) -> tuple[Student, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = Student(Config.from_dict(saved["config"]))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device), saved


def check_compatible(c: Config, metadata: dict, tokenizer: Path, saved: dict | None = None) -> None:
    prepared = Config.from_dict(metadata["config"])
    architecture = (
        "vocab_size",
        "hidden_size",
        "n_layers",
        "n_heads",
        "intermediate_size",
        "output_size",
        "max_len",
        "pad_id",
        "rope_theta",
        "norm_eps",
    )
    if any(getattr(c, name) != getattr(prepared, name) for name in architecture):
        raise ValueError("Prepared data and checkpoint have different model configurations")
    actual = file_hash(tokenizer)
    if actual != metadata["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer has changed")
    if saved and saved.get("tokenizer_sha256") and saved["tokenizer_sha256"] != actual:
        raise ValueError("Checkpoint and data use different tokenizers")
    if saved and not saved.get("tokenizer_sha256") and actual != file_hash(DEFAULT_TOKENIZER):
        raise ValueError("Legacy checkpoint requires its packaged production tokenizer")


def save_training(
    path: Path,
    model: Student,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
    state: dict,
    settings: dict,
) -> None:
    atomic_save(
        {
            "format": 1,
            "model": model.state_dict(),
            "config": asdict(model.config),
            "optimizer": optimizer.state_dict(),
            "dataset_id": metadata["dataset_id"],
            "tokenizer_sha256": metadata["tokenizer_sha256"],
            "state": state,
            "step": state["step"],
            "settings": settings,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        path,
    )
