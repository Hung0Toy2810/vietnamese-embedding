"""One offline preparation pass: clean, tokenize and split retrieval pairs."""

from collections import Counter
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from tokenizers import Tokenizer

from config import Config
from sources import arguments, local_rows, source_rows
from text import InvalidPair, adapt, check_text
from utils import DEFAULT_TOKENIZER, digest, file_hash, write_json


class Components:
    """Queries, known positives and QA contexts must stay in the same split."""

    def __init__(self) -> None:
        self.parent = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left: str, right: str) -> None:
        a, b = sorted((self.find(left), self.find(right)))
        self.parent[b] = a


def prepare(
    rows,
    output: Path,
    tokenizer_path: Path,
    c: Config,
    validation_fraction: float = 0.05,
    max_rows: int = 200000,
) -> dict:
    if not 0 < validation_fraction < 1 or max_rows < 1:
        raise ValueError("Invalid split fraction or row limit")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; reuse it or choose a new output")
    temporary = output.with_name(output.name + ".preparing")
    temporary.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    if tokenizer.get_vocab_size() != c.vocab_size or tokenizer.token_to_id("[PAD]") != c.pad_id:
        raise ValueError("Tokenizer vocabulary or padding ID does not match the model")
    db = sqlite3.connect(temporary / "dataset.sqlite")
    db.executescript("""
        CREATE TABLE texts (id INTEGER PRIMARY KEY, hash TEXT UNIQUE, text TEXT,
                            tokens BLOB, split TEXT);
        CREATE TABLE pairs (id INTEGER PRIMARY KEY, query INTEGER, positive INTEGER,
                            negatives TEXT, source TEXT, split TEXT,
                            UNIQUE(query, positive, negatives));
        CREATE TABLE positives (query INTEGER, document INTEGER, PRIMARY KEY(query, document));
    """)
    counts, groups = Counter(), Components()

    def register(text: str) -> tuple[int, str]:
        check_text(text)
        key = digest(text)
        found = db.execute("SELECT id FROM texts WHERE hash=?", (key,)).fetchone()
        if found:
            return found[0], key
        ids = tokenizer.encode(text).ids
        if not ids or not any(i != c.pad_id for i in ids) or len(ids) > c.max_len or "\n\n" in text:
            raise InvalidPair("overlong_or_multiple_paragraphs")
        blob = np.asarray(ids, dtype=np.int32).tobytes()
        cursor = db.execute("INSERT INTO texts(hash,text,tokens) VALUES(?,?,?)", (key, text, blob))
        text_id = cursor.lastrowid
        if text_id is None:
            raise RuntimeError("Database did not return the inserted text ID")
        return text_id, key

    try:
        for raw in itertools.islice(rows, max_rows):
            counts["raw_rows"] += 1
            try:
                pair = adapt(raw, tokenizer, c)
                query, qkey = register(pair["query"])
                positive, pkey = register(pair["positive"])
                negatives = []
                for text in pair["hard_negatives"]:
                    try:
                        negatives.append(register(text)[0])
                    except InvalidPair:
                        counts["invalid_negative"] += 1
                groups.join(qkey, pkey)
                if pair.get("split_group_id"):
                    groups.join(qkey, "context:" + pair["split_group_id"])
                cursor = db.execute(
                    "INSERT OR IGNORE INTO pairs(query,positive,negatives,source) VALUES(?,?,?,?)",
                    (query, positive, json.dumps(sorted(set(negatives))), pair["source"]),
                )
                counts["duplicate_rows"] += int(cursor.rowcount == 0)
                db.execute("INSERT OR IGNORE INTO positives VALUES(?,?)", (query, positive))
            except InvalidPair as error:
                counts[str(error)] += 1
            if counts["raw_rows"] % 10000 == 0:
                db.commit()
                print(f"Prepared {counts['raw_rows']:,} raw rows", flush=True)
        # Assign whole connected groups, so repeated queries/positives cannot leak across splits.
        for text_id, key in db.execute("SELECT id,hash FROM texts"):
            score = int(digest(f"{c.seed}:{groups.find(key)}")[:16], 16) / 2**64
            split = "validation" if score < validation_fraction else "train"
            db.execute("UPDATE texts SET split=? WHERE id=?", (split, text_id))
        db.execute("UPDATE pairs SET split=(SELECT split FROM texts WHERE id=pairs.query)")
        db.commit()
        filter_pairs(db)
        counts.update(
            {split: n for split, n in db.execute("SELECT split,COUNT(*) FROM pairs GROUP BY split")}
        )
        if not counts["train"]:
            raise ValueError("No training pairs remain after preparation")
        if not counts["validation"]:
            print(
                "Warning: no validation pairs; use more independent groups or a larger split fraction."
            )
        db.executescript(
            "CREATE INDEX pair_split ON pairs(split); CREATE INDEX text_split ON texts(split);"
        )
        db.commit()
    finally:
        db.close()
    shutil.copyfile(tokenizer_path, temporary / "tokenizer.json")
    metadata = {
        "format": 2,
        "config": asdict(c),
        "counts": dict(counts),
        "validation_fraction": validation_fraction,
        "tokenizer_sha256": file_hash(tokenizer_path),
        "database_sha256": file_hash(temporary / "dataset.sqlite"),
    }
    metadata["dataset_id"] = digest(json.dumps(metadata, sort_keys=True))
    write_json(temporary / "metadata.json", metadata)
    temporary.rename(output)
    print(json.dumps(dict(counts), indent=2))
    return metadata


def filter_pairs(db: sqlite3.Connection) -> None:
    splits = dict(db.execute("SELECT id,split FROM texts"))
    for row_id, query, negatives, split in db.execute("SELECT id,query,negatives,split FROM pairs"):
        known = {
            row[0] for row in db.execute("SELECT document FROM positives WHERE query=?", (query,))
        }
        selected = [i for i in json.loads(negatives) if i not in known and splits[i] == split]
        db.execute("UPDATE pairs SET negatives=? WHERE id=?", (json.dumps(selected), row_id))
    db.commit()


def main() -> None:
    parser = arguments()
    parser.add_argument("--output", type=Path, default=Path("data/retrieval"))
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Skip preparation or use a different directory.")
    rows = (
        source_rows(args.sources, Path("data/downloads"))
        if args.sources
        else itertools.chain.from_iterable(local_rows(path) for path in args.input)
    )

    prepare(
        rows,
        args.output,
        args.tokenizer,
        Config(),
        args.validation_fraction,
        args.max_rows,
    )


if __name__ == "__main__":
    main()
