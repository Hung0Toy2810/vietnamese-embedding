"""Context limits are checked between completed inference turns."""

import unittest
from unittest.mock import patch

from legal_rag.budget import ContextBudgetExceeded, ensure_context
from legal_rag.settings import Settings


class ContextTests(unittest.TestCase):
    def test_context_reserves_room_for_generation_and_wrappers(self):
        with patch("legal_rag.budget.token_count", return_value=100):
            estimate = ensure_context(
                [{"role": "user", "content": "short"}],
                [],
                Settings(context_tokens=4096),
            )
        self.assertGreater(estimate, 100)

    def test_oversized_input_is_rejected_before_inference(self):
        with patch("legal_rag.budget.token_count", return_value=4000):
            with self.assertRaises(ContextBudgetExceeded):
                ensure_context(
                    [{"role": "user", "content": "long"}],
                    [],
                    Settings(context_tokens=4096),
                )
