"""Encode short Vietnamese texts with the packaged or a newly fine-tuned model."""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from checkpoint import load_model
from grad_cache import encode
from utils import DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER, file_hash


def embed(
    model, tokenizer: Tokenizer, texts: list[str], device: torch.device, batch_size: int = 32
) -> torch.Tensor:
    tokenizer.no_truncation()
    tokenizer.no_padding()
    records = [{"ids": tokenizer.encode(text).ids} for text in texts]
    if not records or any(
        not row["ids"] or len(row["ids"]) > model.config.max_len for row in records
    ):
        raise ValueError(
            f"Each text must contain 1-{model.config.max_len} tokens; split at complete sentences first"
        )
    with torch.inference_mode():
        return encode(model, records, device, batch_size, bf16=False).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model, saved = load_model(args.checkpoint, device)
    expected = saved.get("tokenizer_sha256", file_hash(DEFAULT_TOKENIZER))
    if file_hash(args.tokenizer) != expected:
        parser.error("Tokenizer does not match this checkpoint")
    model.eval().requires_grad_(False)
    vectors = embed(model, Tokenizer.from_file(str(args.tokenizer)), args.texts, device)
    if args.output:
        if args.output.exists() or args.output.resolve().is_relative_to(
            DEFAULT_CHECKPOINT.parent.resolve()
        ):
            parser.error("Refusing to overwrite existing files or write into checkpoints/")
        torch.save(vectors, args.output)
    print(
        json.dumps(
            {
                "checkpoint_step": saved["step"],
                "shape": list(vectors.shape),
                "l2_norms": vectors.norm(dim=1).tolist(),
                "pairwise_cosine": (vectors @ vectors.T).tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
