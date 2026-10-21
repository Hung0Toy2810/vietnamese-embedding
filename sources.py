"""Read local JSONL/Parquet or download a finite list of pinned Parquet files."""

import argparse
import itertools
import json
from pathlib import Path


def local_rows(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON in {path}:{number}") from error
    elif path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        for batch in parquet.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        raise ValueError("Input must be .jsonl or .parquet")


def source_rows(manifest: Path, cache: Path):
    from huggingface_hub import hf_hub_download

    specs = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
    for spec in specs:
        if len(spec["revision"]) != 40 or not spec["files"]:
            raise ValueError("Each source needs a pinned revision and explicit files")
        # Every selected file is fully downloaded before it is read. No HTTP training stream.
        paths = [
            Path(
                hf_hub_download(
                    spec["dataset"],
                    name,
                    repo_type="dataset",
                    revision=spec["revision"],
                    cache_dir=str(cache),
                )
            )
            for name in spec["files"]
        ]
        rows = itertools.chain.from_iterable(local_rows(path) for path in paths)
        for row in itertools.islice(rows, spec["max_rows"]):
            yield {**row, "source": spec["dataset"], "adapter": spec["adapter"]}


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, nargs="+", help="Local JSONL/Parquet files")
    source.add_argument("--sources", type=Path, help="Pinned offline download manifest")
    return parser
