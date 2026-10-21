"""Train retrieval from a prepared folder. No downloads or tokenization during training."""

import argparse
import csv
import math
from pathlib import Path
import signal

import torch

from checkpoint import check_compatible, load_model, save_training
from data import Corpus
from grad_cache import cached_backward
from losses import retrieval_loss
from model import Student
from utils import DEFAULT_CHECKPOINT, ROOT, file_hash


def train(
    data: Path,
    output: Path,
    settings: dict,
    device: str | torch.device = "cpu",
    init: Path | None = DEFAULT_CHECKPOINT,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict:
    device = torch.device(device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        budget = min(18e9, 0.9 * total, free - 0.5e9)
        if budget <= 0:
            raise RuntimeError("Not enough free GPU memory")
        torch.cuda.set_per_process_memory_fraction(budget / total, device)
    protected = ROOT / "checkpoints"
    if output.resolve() == ROOT or output.resolve().is_relative_to(protected.resolve()):
        raise ValueError("Choose a new run directory outside checkpoints/")
    if output.resolve() == data.resolve() or output.resolve().is_relative_to(data.resolve()):
        raise ValueError("Run output must be separate from prepared data")
    if init is not None and init.resolve().is_relative_to(output.resolve()):
        raise ValueError("Initialization checkpoint must be outside the output directory")
    if any(
        settings[key] < 1 for key in ("epochs", "batch_size", "micro_batch", "checkpoint_every")
    ):
        raise ValueError("Epoch, batch and checkpoint counts must be positive")
    if settings["warmup_steps"] < 0 or settings["lr"] <= 0:
        raise ValueError("Invalid learning rate or warmup")
    if stop_after is not None and stop_after < 1:
        raise ValueError("Stop-after must be a positive update count")
    corpus = Corpus(data)
    saved, state = None, {"epoch": 0, "next_batch": 0, "step": 0}
    try:
        torch.manual_seed(corpus.config.seed)
        if resume:
            model, saved = load_model(output / "latest.pt", device)
            if saved.get("format") != 1 or "optimizer" not in saved:
                raise ValueError("Resume needs a new training checkpoint with optimizer state")
            if (
                saved["dataset_id"] != corpus.metadata["dataset_id"]
                or saved["settings"] != settings
            ):
                raise ValueError("Resume requires the same prepared dataset and training settings")
            state = saved["state"]
        else:
            if output.exists():
                raise FileExistsError("Run already exists; use --resume or a new output directory")
            if init is None:
                model = Student(corpus.config).to(device)
            else:
                model, saved = load_model(init, device)
        check_compatible(model.config, corpus.metadata, data / "tokenizer.json", saved)
        if file_hash(data / "dataset.sqlite") != corpus.metadata["database_sha256"]:
            raise ValueError("Prepared dataset has changed; prepare into a new folder")
        if device.type == "cuda" and settings["bf16"] and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support BF16; pass --fp32")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["lr"], betas=(0.9, 0.95), weight_decay=0.01
        )
        if resume:
            if saved is None:
                raise RuntimeError("Resume checkpoint was not loaded")
            optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["rng"])
            if device.type == "cuda" and saved["cuda_rng"]:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        output.mkdir(parents=True, exist_ok=True)
        model.train()
        return run_epochs(model, optimizer, corpus, output, state, settings, device, stop_after)
    finally:
        corpus.close()


def run_epochs(
    model,
    optimizer,
    corpus: Corpus,
    output: Path,
    state: dict,
    settings: dict,
    device: torch.device,
    stop_after: int | None,
) -> dict:
    if stop_after is not None and state["step"] >= stop_after:
        return state
    stopped = False

    def request_stop(signum, frame) -> None:
        nonlocal stopped
        stopped = True
        print("Stop requested; saving after the current optimizer update.", flush=True)

    rows = corpus.row_ids("train")
    if not rows:
        raise ValueError("No training rows")
    log_path = output / "metrics.csv"
    columns = ["step", "epoch", "loss", "lr", "grad_norm"]
    if log_path.exists():
        with log_path.open() as stream:
            history = [row for row in csv.DictReader(stream) if int(row["step"]) <= state["step"]]
        with log_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(history)
    handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with log_path.open("a", newline="") as log:
            writer = csv.DictWriter(log, fieldnames=columns)
            if log.tell() == 0:
                writer.writeheader()
            for epoch in range(state["epoch"], settings["epochs"]):
                generator = torch.Generator().manual_seed(corpus.config.seed + epoch)
                order = torch.randperm(len(rows), generator=generator).tolist()
                batch_count = math.ceil(len(rows) / settings["batch_size"])
                start_batch = state["next_batch"] if epoch == state["epoch"] else 0
                for batch in range(start_batch, batch_count):
                    selected = order[
                        batch * settings["batch_size"] : (batch + 1) * settings["batch_size"]
                    ]
                    records, n, positives = corpus.batch([rows[i] for i in selected], device)
                    lr = settings["lr"] * min(
                        1.0, (state["step"] + 1) / max(1, settings["warmup_steps"])
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    loss_fn = lambda vectors: retrieval_loss(
                        vectors,
                        n,
                        positives,
                        model.config.temperature,
                    )
                    metrics = cached_backward(
                        model, records, loss_fn, device, settings["micro_batch"], settings["bf16"]
                    )
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    state = {"epoch": epoch, "next_batch": batch + 1, "step": state["step"] + 1}
                    writer.writerow(
                        {
                            "step": state["step"],
                            "epoch": epoch + 1,
                            **metrics,
                            "lr": lr,
                            "grad_norm": float(norm),
                        }
                    )
                    log.flush()
                    should_stop = stopped or (
                        stop_after is not None and state["step"] >= stop_after
                    )
                    if state["step"] % settings["checkpoint_every"] == 0 or should_stop:
                        save_training(
                            output / "latest.pt", model, optimizer, corpus.metadata, state, settings
                        )
                    if state["step"] % 10 == 0 or should_stop:
                        print(
                            f"step={state['step']} loss={metrics['loss']:.4f}",
                            flush=True,
                        )
                    if should_stop:
                        return state
                state = {**state, "epoch": epoch + 1, "next_batch": 0}
                save_training(
                    output / "latest.pt", model, optimizer, corpus.metadata, state, settings
                )
        return state
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/retrieval"))
    parser.add_argument("--output", type=Path, default=Path("runs/retrieval"))
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64, help="Queries per effective batch")
    parser.add_argument("--micro-batch", type=int, default=16, help="Sequences per encoder pass")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--stop-after", type=int, help="Pause after this many total updates; resumable"
    )
    args = parser.parse_args()
    settings = {
        key: getattr(args, key)
        for key in ("epochs", "batch_size", "micro_batch", "lr", "warmup_steps", "checkpoint_every")
    }
    settings.update(bf16=not args.fp32)
    state = train(
        args.data,
        args.output,
        settings,
        args.device,
        None if args.from_scratch else args.init_checkpoint,
        args.resume,
        args.stop_after,
    )
    print(f"Saved {args.output / 'latest.pt'} at step {state['step']}")


if __name__ == "__main__":
    main()
