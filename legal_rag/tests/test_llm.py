"""Stream protocol tests; no live model calls are made."""

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from legal_rag.llm import chat, structured
from legal_rag.schemas import ChunkPlan
from legal_rag.settings import Settings


def stream(events: list[dict]) -> io.BytesIO:
    lines = [json.dumps(event).encode() + b"\n" for event in events]
    return io.BytesIO(b"".join(lines))


class StreamTests(unittest.TestCase):
    def test_answer_profile_reaches_native_json_generation(self):
        with (
            patch("legal_rag.llm.ensure_context"),
            patch(
                "legal_rag.llm.chat", return_value='{"rationale":"Fixture"}'
            ) as request,
        ):
            structured("Fixture", ChunkPlan, Settings(), answer=True)
        payload = request.call_args.args[0]
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["temperature"], 0.7)
        self.assertEqual(payload["options"]["top_p"], 0.8)
        self.assertEqual(payload["options"]["presence_penalty"], 1.5)
        self.assertEqual(payload["options"]["seed"], 42)
        self.assertEqual(payload["options"]["num_predict"], -1)

    def test_structured_extraction_disables_inherited_repetition_penalties(self):
        with (
            patch("legal_rag.llm.ensure_context"),
            patch(
                "legal_rag.llm.chat", return_value='{"rationale":"Fixture"}'
            ) as request,
        ):
            structured("Fixture input", ChunkPlan, Settings())
        payload = request.call_args.args[0]
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["presence_penalty"], 0)
        self.assertEqual(payload["options"]["frequency_penalty"], 0)
        self.assertEqual(payload["options"]["repeat_penalty"], 1)
        self.assertEqual(payload["options"]["num_predict"], -1)

    def test_explicit_nonreasoning_mode_still_waits_for_complete_stream(self):
        events = [{"message": {"content": "{}"}, "done": True, "done_reason": "stop"}]
        with patch("legal_rag.llm.urlopen", return_value=stream(events)) as request:
            self.assertEqual(
                chat({"model": "qwen3.5:9b", "think": False}, "http://localhost:11434"),
                "{}",
            )
        self.assertFalse(json.loads(request.call_args.args[0].data)["think"])

    def test_waits_through_thinking_and_joins_complete_answer(self) -> None:
        events = [
            {"message": {"thinking": "Đang phân tích câu hỏi"}, "done": False},
            {"message": {"content": "Đáp "}, "done": False},
            {"message": {"content": "án."}, "done": True, "done_reason": "stop"},
        ]
        payload = {"model": "qwen3.5:9b", "options": {"num_predict": -1}}
        output = io.StringIO()
        with patch("legal_rag.llm.urlopen", return_value=stream(events)) as request:
            with patch("legal_rag.llm.time.monotonic", side_effect=[0, 21, 42]):
                with redirect_stdout(output):
                    self.assertEqual(chat(payload, "http://localhost:11434"), "Đáp án.")
        self.assertIsNone(request.call_args.kwargs["timeout"])
        sent = json.loads(request.call_args.args[0].data)
        self.assertTrue(sent["think"])
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["options"]["num_predict"], -1)
        self.assertNotIn("Đang phân tích câu hỏi", output.getvalue())

    def test_disconnect_is_not_reported_as_completed_answer(self) -> None:
        with patch(
            "legal_rag.llm.urlopen",
            return_value=stream(
                [{"message": {"content": "Đáp án còn dang dở"}, "done": False}]
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "before Ollama"):
                chat({"model": "qwen3.5:9b"}, "http://localhost:11434")

    def test_token_limit_is_not_reported_as_completed_answer(self) -> None:
        with patch(
            "legal_rag.llm.urlopen",
            return_value=stream(
                [
                    {
                        "message": {"content": "Đang viết"},
                        "done": True,
                        "done_reason": "length",
                    }
                ]
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "token limit"):
                chat({"model": "qwen3.5:9b"}, "http://localhost:11434")

    def test_server_error_is_not_silently_ignored(self) -> None:
        with patch(
            "legal_rag.llm.urlopen", return_value=stream([{"error": "model failed"}])
        ):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                chat({"model": "qwen3.5:9b"}, "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()
