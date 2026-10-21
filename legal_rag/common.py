"""Source-preserving text windows and atomic JSON persistence."""

import json
import os
import re
from bisect import bisect_right
from pathlib import Path

from tokenizers import Tokenizer

DATA = Path(__file__).resolve().parent / "data"
RERANKER = "BAAI/bge-reranker-v2-m3"


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def token_ranges(text: str, tokenizer: Tokenizer, limit: int) -> list[tuple[int, int]]:
    """Prefer sentence endings; split oversized sentences by original offsets.

    Whitespace and page breaks do not become semantic boundaries. Returned ranges
    cover every non-whitespace character without decoding or rewriting source text.
    """
    if limit < 1:
        raise ValueError("The token limit must be positive.")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    token_ends = [end for _start, end in encoded.offsets]
    ranges = []
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text):
            break
        cursor = bisect_right(token_ends, start)
        if cursor >= len(token_ends):
            raise ValueError("The tokenizer cannot represent part of the source text.")
        if len(token_ends) - cursor <= limit:
            end = len(text)
        else:
            end = token_ends[cursor + limit - 1]
            endings = list(
                re.finditer(r"[.!?;。！？](?:[\"'”’)]*)\s+", text[start:end])
            )
            if endings and endings[-1].end() > (end - start) // 3:
                end = start + endings[-1].end()
        if end <= start:
            raise ValueError("Tokenization did not advance through the source text.")
        while len(tokenizer.encode(text[start:end]).ids) > limit:
            end -= 1
        if end <= start:
            raise ValueError("A source character exceeds the token budget.")
        ranges.append((start, end))
        start = end
    return ranges


def split_child(
    text: str, tokenizer: Tokenizer, limit: int = 224, overlap: int = 0
) -> list[str]:
    if overlap:
        raise ValueError(
            "Use graph neighbors for context expansion instead of duplicated token overlap."
        )
    return [text[start:end] for start, end in token_ranges(text, tokenizer, limit)]
