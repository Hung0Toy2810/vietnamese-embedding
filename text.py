"""Normalize Vietnamese and keep complete, answer-bearing QA passages."""

import re
import unicodedata
from collections.abc import Iterator
from typing import Any

from tokenizers import Tokenizer

from config import Config
from utils import digest


def clean(text: str) -> str:
    if not isinstance(text, str):
        raise InvalidPair("non_text_value")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(
        r"</?(?:p|div|li|section|article|h[1-6])\b[^>]*>|<br\b[^>]*>", "\n\n", text, flags=re.I
    )
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]
    return "\n\n".join(line for line in lines if line)


def sentences(text: str) -> Iterator[str]:
    start = 0
    abbreviations = {"tp.", "ts.", "ths.", "pgs.", "gs.", "bs.", "mr.", "mrs.", "dr.", "v.v."}
    for boundary in re.finditer(r"[.!?…]+[\"'”’»\)\]]*\s+", text):
        prefix = text[start : boundary.end()].rstrip()
        if prefix.split()[-1].lower() not in abbreviations:
            yield prefix
            start = boundary.end()
    if text[start:].strip():
        yield text[start:].strip()


class InvalidPair(ValueError):
    pass


def check_text(text: str) -> None:
    bad = ("\ufffd", "\x00", "Ãƒ", "Ã‚", "Ã¡", "Ã©", "Ã³", "Â¢", "Â£", "â€", "ðŸ")
    if any(marker in text for marker in bad):
        raise InvalidPair("corrupt_encoding")
    if len(text.strip()) < 3 or sum(ch.isalnum() for ch in text) / max(1, len(text)) < 0.25:
        raise InvalidPair("empty_or_noise")


def squad_pair(row: dict[str, Any], tokenizer: Tokenizer, c: Config) -> dict[str, Any]:
    if row.get("is_impossible") in (True, "True", "true", 1):
        raise InvalidPair("unanswerable")
    context: str = row["context"]
    raw_answers = row.get("answers", row.get("answer", {})) or {}
    answers: dict[str, Any]
    if isinstance(raw_answers, str):
        answers = {"text": [raw_answers]}
    elif isinstance(raw_answers, list):
        answers = {"text": raw_answers}
    elif isinstance(raw_answers, dict):
        answers = raw_answers
    else:
        raise InvalidPair("invalid_answer_format")
    texts = answers.get("text", [])
    starts = answers.get("answer_start", [])
    if not texts:
        raise InvalidPair("no_answer")
    # Locate a verified span in the original context before cleaning or sentence packing.
    found: tuple[int, int, str] | None = None
    for k, answer in enumerate(texts):
        if not answer:
            continue
        raw_start = starts[k] if k < len(starts) else -1
        start = raw_start if isinstance(raw_start, int) else -1
        if start >= 0 and context[start : start + len(answer)].casefold() == answer.casefold():
            found = start, start + len(answer), answer
            break
        matches = list(re.finditer(re.escape(answer), context, re.IGNORECASE))
        if len(matches) == 1:
            found = matches[0].start(), matches[0].end(), matches[0].group()
            break
    if found is None:
        raise InvalidPair("answer_span_unverified")
    start, end, answer = found
    offset = 0
    for paragraph_index, paragraph in enumerate(context.split("\n\n")):
        if offset <= start and end <= offset + len(paragraph):
            break
        offset += len(paragraph) + 2
    else:
        raise InvalidPair("answer_crosses_paragraph")
    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    for sentence in sentences(paragraph):
        sentence = sentence.strip()
        pos = paragraph.find(sentence, cursor)
        pieces.append((sentence, pos, pos + len(sentence)))
        cursor = pos + len(sentence)
    selected = [i for i, (_, a, b) in enumerate(pieces) if a < end - offset and b > start - offset]
    if not selected:
        raise InvalidPair("answer_sentence_missing")
    left, right = min(selected), max(selected)

    def passage(lo: int, hi: int) -> str:
        return clean(" ".join(p[0] for p in pieces[lo : hi + 1]))

    positive = passage(left, right)
    size = len(tokenizer.encode(positive).ids)
    if size > c.max_len:
        raise InvalidPair("answer_sentence_too_long")
    target = c.chunk_targets[int(digest(context + row["question"])[:8], 16) % len(c.chunk_targets)]
    while size < target:
        candidates: list[tuple[int, int, int, str, int]] = []
        for lo, hi in ((left - 1, right), (left, right + 1)):
            if lo < 0 or hi >= len(pieces):
                continue
            value = passage(lo, hi)
            count = len(tokenizer.encode(value).ids)
            if count <= c.max_len:
                candidates.append((abs(count - target), lo, hi, value, count))
        if not candidates:
            break
        _, lo, hi, value, count = min(candidates)
        if abs(count - target) >= abs(size - target):
            break
        left, right, positive, size = lo, hi, value, count
    if clean(answer).casefold() not in positive.casefold():
        raise InvalidPair("answer_lost_after_cleaning")
    return {
        "query": clean(row["question"]),
        "positive": positive,
        "hard_negatives": [],
        "split_group_id": digest(clean(context)),
        "paragraph_index": paragraph_index,
        "answer_text": clean(answer),
        "source": row.get("source", "local"),
    }


def adapt(row: dict[str, Any], tokenizer: Tokenizer, c: Config) -> dict[str, Any]:
    adapter = row.get("adapter", "squad_answer_v1" if "context" in row else "triples_v1")
    if adapter == "squad_answer_v1":
        pair = squad_pair(row, tokenizer, c)
    elif adapter == "triples_v1":
        pair = {
            "query": clean(row["query"]),
            "positive": clean(row["positive"]),
            "hard_negatives": row.get(
                "hard_negatives", [row["negative"]] if row.get("negative") else []
            ),
            "source": row.get("source", "local"),
        }
    else:
        raise InvalidPair(f"unknown_adapter:{adapter}")
    check_text(pair["query"])
    check_text(pair["positive"])
    if not isinstance(pair["hard_negatives"], (list, tuple)):
        raise InvalidPair("negatives_must_be_a_list")
    negatives = []
    for text in pair["hard_negatives"]:
        try:
            text = clean(text)
            check_text(text)
        except InvalidPair:
            continue
        if text != pair["positive"] and text not in negatives:
            negatives.append(text)
    pair["hard_negatives"] = negatives
    return pair
