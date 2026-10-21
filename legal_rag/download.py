"""Download a sample PDF as-is, without document-specific text repairs."""

import argparse
from pathlib import Path
from urllib.request import urlopen

SAMPLE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SAMPLE_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "sample.pdf",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The output already exists; refusing to overwrite it.")
    with urlopen(args.url, timeout=120) as response:
        content = response.read(25 * 1024 * 1024 + 1)
    if not content.startswith(b"%PDF") or len(content) > 25 * 1024 * 1024:
        raise ValueError("The source is not a PDF or exceeds 25 MB.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(
        f"Saved {args.output}. Ingest it with: python -m legal_rag.build {args.output}"
    )


if __name__ == "__main__":
    main()
