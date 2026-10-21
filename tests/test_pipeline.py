"""CPU regression tests for the simplified retrieval pipeline; no downloads."""

import copy
from dataclasses import asdict, replace
import json
import signal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from data import Corpus, padded
from evaluate import evaluate, ranking_metrics, search
from grad_cache import cached_backward
from losses import retrieval_loss
from model import Student
from prepare import prepare
from text import InvalidPair, adapt, squad_pair
from train import train
from utils import file_hash


def tiny_config() -> Config:
    return replace(
        Config(),
        vocab_size=96,
        hidden_size=24,
        n_layers=2,
        n_heads=3,
        intermediate_size=40,
        output_size=16,
        max_len=32,
        chunk_targets=(8, 16, 32),
    )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.c = tiny_config()
        vocab = {"[PAD]": 0, "[UNK]": 1, **{f"t{i}": i + 2 for i in range(94)}}
        self.tokenizer = Tokenizer.from_str(
            json.dumps(
                {
                    "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
                    "pre_tokenizer": {"type": "WhitespaceSplit"},
                }
            )
        )
        self.tokenizer_path = self.root / "tokenizer.json"
        self.tokenizer.save(str(self.tokenizer_path))

    def tearDown(self):
        self.temp.cleanup()

    def rows(self):
        return [
            {
                "query": f"t{i} câu hỏi là gì",
                "positive": f"t{i} nội dung trả lời.",
                "negative": f"t{i + 40} văn bản khác.",
            }
            for i in range(40)
        ]

    def build(self, rows=None):
        output = self.root / "prepared"
        prepare(
            self.rows() if rows is None else rows,
            output,
            self.tokenizer_path,
            self.c,
            validation_fraction=0.3,
        )
        return output

    def test_answer_and_text_rules(self):
        context = "Một đoạn khác.\n\nThủ đô là Hà Nội. Thành phố có nhiều hồ đẹp."
        row = {
            "question": "Thủ đô là gì?",
            "context": context,
            "answers": {"text": ["Hà Nội"], "answer_start": [context.index("Hà Nội")]},
        }
        result = squad_pair(row, self.tokenizer, self.c)
        self.assertIn("Hà Nội", result["positive"])
        self.assertNotIn("Một đoạn khác", result["positive"])
        self.assertNotIn("\n\n", result["positive"])
        for changed in (
            {**row, "is_impossible": True},
            {**row, "answers": {"text": ["không có"]}},
            {**row, "context": "từ " * 40 + "Hà Nội.", "answers": {"text": ["Hà Nội"]}},
        ):
            with self.assertRaises(InvalidPair):
                squad_pair(changed, self.tokenizer, self.c)
        with self.assertRaises(InvalidPair):
            adapt({"query": "lỗi mã hóa ÃƒÂ", "positive": "Nội dung đúng."}, self.tokenizer, self.c)

    def test_split_labels_and_prepared_data(self):
        rows = self.rows()
        rows += [
            rows[0],
            {**rows[0], "positive": rows[1]["positive"], "negative": rows[0]["positive"]},
        ]
        output = self.build(rows)
        corpus = Corpus(output)
        self.assertGreater(len(corpus.row_ids("train")), 0)
        self.assertGreater(len(corpus.row_ids("validation")), 0)
        splits = {i: s for i, s in corpus.db.execute("SELECT id,split FROM texts")}
        for q, p, negatives, split in corpus.db.execute(
            "SELECT query,positive,negatives,split FROM pairs"
        ):
            self.assertEqual(splits[q], split)
            self.assertEqual(splits[p], split)
            for negative in json.loads(negatives):
                self.assertEqual(splits[negative], split)
                self.assertNotIn(negative, corpus.relevant(q))
        self.assertEqual(corpus.metadata["counts"]["duplicate_rows"], 1)
        # Repeated query has both answers, including the one supplied as a negative elsewhere.
        q = corpus.db.execute(
            "SELECT query FROM positives GROUP BY query HAVING COUNT(*)=2"
        ).fetchone()[0]
        ids = [i for (i,) in corpus.db.execute("SELECT id FROM pairs WHERE query=?", (q,))]
        records, n, mask = corpus.batch(ids, torch.device("cpu"))
        self.assertTrue((mask.sum(dim=1) == 2).all())
        self.assertEqual(len(mask), n)
        corpus.close()
        before = file_hash(output / "dataset.sqlite")
        with self.assertRaises(FileExistsError):
            prepare([], output, self.tokenizer_path, self.c)
        self.assertEqual(before, file_hash(output / "dataset.sqlite"))

    def test_cached_gradients_match_full_batch(self):
        torch.manual_seed(7)
        full = Student(self.c)
        records = [{"ids": [2, 3, 4, 5][: 1 + i % 4]} for i in range(11)]
        mask = torch.zeros(3, 8, dtype=torch.bool)
        mask[torch.arange(3), torch.arange(3)] = True
        mask[0, 1] = True
        cached = copy.deepcopy(full)
        loss_fn = lambda value: retrieval_loss(value, 3, mask)
        ids, attention = padded(records, self.c, torch.device("cpu"))
        loss = loss_fn(full(ids, attention))
        loss.backward()
        result = cached_backward(cached, records, loss_fn, torch.device("cpu"), 4)
        self.assertAlmostEqual(float(loss.detach()), result["loss"], places=5)
        for left, right in zip(full.parameters(), cached.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=3e-5, rtol=3e-4)

    def test_offline_train_resume_and_evaluate(self):
        output = self.build()
        initial = self.root / "initial.pt"
        torch.manual_seed(123)
        torch.save(
            {
                "config": asdict(self.c),
                "model": Student(self.c).state_dict(),
                "step": 0,
                "tokenizer_sha256": file_hash(self.tokenizer_path),
            },
            initial,
        )
        initial_hash = file_hash(initial)
        settings = {
            "epochs": 2,
            "batch_size": 5,
            "micro_batch": 4,
            "lr": 0.001,
            "warmup_steps": 3,
            "checkpoint_every": 2,
            "bf16": False,
        }
        with patch("sources.source_rows", side_effect=AssertionError("Training accessed network")):
            full_state = train(output, self.root / "full", settings, init=initial)
            train(output, self.root / "resumed", settings, init=initial, stop_after=2)
            resumed_state = train(
                output, self.root / "resumed", settings, init=initial, resume=True
            )
        self.assertEqual(full_state, resumed_state)
        full = torch.load(self.root / "full/latest.pt", weights_only=True)
        resumed = torch.load(self.root / "resumed/latest.pt", weights_only=True)
        for key in full["model"]:
            torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=0, rtol=0)
        self.assertEqual(
            (self.root / "full/metrics.csv").read_text(),
            (self.root / "resumed/metrics.csv").read_text(),
        )
        self.assertEqual(file_hash(initial), initial_hash)
        result = evaluate(output, self.root / "full/latest.pt", batch_size=4)
        self.assertGreater(result["query_count"], 0)
        self.assertTrue(0 <= result["metrics"]["mrr10"] <= 1)
        with self.assertRaises(FileExistsError):
            train(output, self.root / "full", settings, init=initial)
        with self.assertRaises(ValueError):
            train(
                output, self.root / "resumed", {**settings, "lr": 0.002}, init=initial, resume=True
            )

    def test_loss_and_metrics(self):
        output = self.build()
        corpus = Corpus(output)
        rows = corpus.row_ids("train")[:3]
        records, n, mask = corpus.batch(rows, torch.device("cpu"))
        ids, attention = padded(records, self.c, torch.device("cpu"))
        loss = retrieval_loss(Student(self.c)(ids, attention), n, mask)
        self.assertTrue(torch.isfinite(loss))
        corpus.close()
        result = ranking_metrics([[1, 3, 2], [9, 2]], [{1, 2}, {2}])
        self.assertEqual(result["mrr10"], 0.75)
        self.assertEqual(result["hit1"], 0.5)
        self.assertEqual(result["recall1"], 0.25)
        queries, docs = torch.randn(3, 16), torch.randn(23, 16)
        docs[2] = docs[1]
        expected = (queries @ docs.T).argsort(dim=1, descending=True, stable=True)[:, :10]
        torch.testing.assert_close(search(queries, docs, chunk_size=4), expected)

    def test_stop_signal_and_training_guards(self):
        output = self.build()
        settings = {
            "epochs": 1,
            "batch_size": 5,
            "micro_batch": 4,
            "lr": 0.001,
            "warmup_steps": 0,
            "checkpoint_every": 100,
            "bf16": False,
        }
        before = signal.getsignal(signal.SIGTERM)

        def stop_after_backward(*args, **kwargs):
            result = cached_backward(*args, **kwargs)
            signal.raise_signal(signal.SIGTERM)
            return result

        with patch("train.cached_backward", side_effect=stop_after_backward):
            state = train(output, self.root / "signal", settings, init=None)
        self.assertEqual(state["step"], 1)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)
        snapshot = torch.load(self.root / "signal/latest.pt", weights_only=True)
        self.assertEqual(snapshot["state"], state)
        limited = train(
            output, self.root / "signal", settings, init=None, resume=True, stop_after=1
        )
        self.assertEqual(limited, state)
        with self.assertRaises(ValueError):
            train(output, output / "run", settings, init=None)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
