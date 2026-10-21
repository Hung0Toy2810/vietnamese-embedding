"""Source-local evidence validation for arbitrary document subjects."""

import json
import unittest
from legal_rag.grounding import REFUSAL, grounded_answer


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.passages = [
            ["Returns are accepted within seven days."],
            ["Disconnect power before cleaning."],
        ]
        self.answer = {
            "answerable": True,
            "answer": "Returns are accepted within seven days. [Source 1]",
            "evidence": [{"source": 1, "passage": 1, "quote": "within seven days"}],
            "follow_up": [],
        }

    def render(self, value):
        return grounded_answer(json.dumps(value), self.passages)

    def test_valid_generic_source_is_accepted(self):
        self.assertEqual(self.render(self.answer), self.answer["answer"])

    def test_invalid_source_passage_and_quote_are_rejected(self):
        for evidence in (
            {"source": 3, "passage": 1, "quote": "within seven days"},
            {"source": 1, "passage": 2, "quote": "within seven days"},
            {"source": 1, "passage": 1, "quote": "thirty days"},
            {"source": 1, "passage": 1, "quote": " "},
        ):
            self.assertEqual(
                self.render({**self.answer, "evidence": [evidence]}), REFUSAL
            )

    def test_a_quote_from_another_source_is_rejected(self):
        evidence = [{"source": 1, "passage": 1, "quote": "Disconnect power"}]
        self.assertEqual(self.render({**self.answer, "evidence": evidence}), REFUSAL)

    def test_fabricated_or_missing_citations_are_rejected(self):
        for text in ("Seven days.", "Seven days. [Source 2]"):
            self.assertEqual(self.render({**self.answer, "answer": text}), REFUSAL)

    def test_unanswerable_and_malformed_outputs_are_rejected(self):
        self.assertEqual(self.render({**self.answer, "answerable": False}), REFUSAL)
        self.assertEqual(grounded_answer("not json", self.passages), REFUSAL)
        self.assertEqual(self.render({**self.answer, "evidence": []}), REFUSAL)

    def test_numeric_boolean_and_unknown_fields_are_rejected(self):
        for value in (
            {**self.answer, "answerable": 1},
            {**self.answer, "unexpected": True},
        ):
            self.assertEqual(self.render(value), REFUSAL)

    def test_translated_citation_label_preserves_source_identity(self):
        answer = {**self.answer, "answer": "Seven days. [Nguồn 1]"}
        self.assertEqual(self.render(answer), "Seven days. [Source 1]")
        self.assertEqual(
            self.render({**answer, "answer": "Seven days. [Nguồn 2]"}), REFUSAL
        )


if __name__ == "__main__":
    unittest.main()
