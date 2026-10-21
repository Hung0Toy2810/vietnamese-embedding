"""Preserve full-batch negatives while replaying only a small encoder microbatch."""

from contextlib import nullcontext
from collections.abc import Callable

import torch
from torch import Tensor

from data import padded
from model import Student


def encode(
    model: Student, records: list[dict], device: torch.device, micro_batch: int, bf16: bool = True
) -> Tensor:
    outputs = []
    for start in range(0, len(records), micro_batch):
        ids, mask = padded(records[start : start + micro_batch], model.config, device)
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and bf16
            else nullcontext()
        )
        with amp:
            outputs.append(model(ids, mask))
    return torch.cat(outputs)


def cached_backward(
    model: Student,
    records: list[dict],
    loss_fn: Callable[[Tensor], Tensor],
    device: torch.device,
    micro_batch: int,
    bf16: bool = True,
) -> dict:
    # This model has no dropout or running statistics, so replay uses the same embeddings.
    with torch.no_grad():
        embeddings = encode(model, records, device, micro_batch, bf16)
    leaf = embeddings.detach().requires_grad_(True)
    loss = loss_fn(leaf)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite loss; no optimizer update was applied")
    loss.backward()
    gradient = leaf.grad
    if gradient is None:
        raise RuntimeError("Loss must depend on the encoded embeddings")
    for start in range(0, len(records), micro_batch):
        part = records[start : start + micro_batch]
        replay = encode(model, part, device, micro_batch, bf16)
        replay.backward(gradient[start : start + len(part)])
    return {"loss": float(loss.detach())}
