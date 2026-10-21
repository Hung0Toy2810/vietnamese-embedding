"""Exercise the real tool loop and source-validated native JSON writer."""

import json
import unittest
from typing import cast
from unittest.mock import patch

import networkx as nx
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from legal_rag.agent import allowed_url, research
from legal_rag.grounding import REFUSAL
from legal_rag.retrieval import Retrieval, relevant_parents
from legal_rag.settings import Settings


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, number=1):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": str(number)}],
        response_metadata={"done": True, "done_reason": "stop"},
    )


def ready(done=True):
    return AIMessage(content="Sources are ready.", response_metadata={"done": done})


PARENT = {
    "id": "p1",
    "document_id": "doc",
    "title": "Store policy",
    "atom_ids": ["a"],
    "passages": ["Returns are accepted within seven days."],
    "text": "Returns are accepted within seven days.",
    "spans": [],
}
ANSWER = {
    "answerable": True,
    "claims": [
        {
            "text": "Returns are accepted within seven days.",
            "references": [{"source": 1, "passage": 1}],
        }
    ],
    "follow_up": [],
    "requested_parts": ["Return deadline"],
    "unsupported_parts": [],
}


class FakeRetrieval:
    def __init__(self, preceding=False):
        self.parents = {"p1": PARENT}
        self.document = {"id": "doc", "name": "Store policy"}
        self.graph = nx.MultiDiGraph()
        if preceding:
            self.graph.add_node("before", text="Policy R-17.", spans=[])
        self.graph.add_node("a", text=PARENT["text"], spans=[], subject="a")

    def search(self, question):
        return [{"parent_id": "p1", "atom_ids": ["a"], "text": PARENT["text"]}]

    def expand(self, children):
        return [PARENT]


class AgentTests(unittest.TestCase):
    def test_subject_selection_separates_matching_words_from_matching_scope(self):
        graph = nx.MultiDiGraph()
        graph.add_node("returns", text="Product return policy.", subject="returns")
        graph.add_node("cleaning", text="Machine cleaning policy.", subject="cleaning")
        parents = [
            {"id": "p1", "atom_ids": ["returns"], "text": "Product return policy."},
            {"id": "p2", "atom_ids": ["cleaning"], "text": "Machine cleaning policy."},
        ]
        for indices, expected in (([1], ["p1"]), ([1, 2], ["p1", "p2"]), ([], [])):
            with (
                patch(
                    "legal_rag.llm.chat", return_value=json.dumps({"groups": indices})
                ),
                patch("legal_rag.llm.ensure_context", return_value=0),
            ):
                chosen = relevant_parents(
                    graph, parents, "What policies apply?", Settings()
                )
            self.assertEqual([p["id"] for p in chosen], expected)

    def run_agent(self, responses=None, outputs=None, settings=None, retrieval=None):
        with (
            patch(
                "legal_rag.llm.chat",
                side_effect=[json.dumps(value) for value in (outputs or [ANSWER])],
            ),
            patch("legal_rag.llm.ensure_context", return_value=0),
        ):
            return research(
                "What is the return deadline?",
                cast(Retrieval, retrieval or FakeRetrieval()),
                settings or Settings(),
                model=ToolModel(responses=responses or [ready()]),
            )

    def test_search_read_and_mindmap_use_real_framework(self):
        result = self.run_agent(
            [
                call("read_section", {"section_id": "p1"}),
                call("create_mindmap", {"source_numbers": [1]}, 2),
                ready(),
            ]
        )
        self.assertIn("[Source 1]", result["answer"])
        self.assertEqual(len(result["mindmap"]["nodes"]), 1)
        self.assertEqual(result["mindmap"]["edges"], [])
        self.assertEqual(result["parents"][0]["subject_context"][0]["anchor"], "a")

    def test_invalid_references_are_repaired_and_original_text_is_copied(self):
        bad = {
            **ANSWER,
            "claims": [
                {"text": "Invented rule", "references": [{"source": 1, "passage": 2}]}
            ],
        }
        result = self.run_agent(outputs=[bad, ANSWER])
        self.assertNotIn("Invented rule", result["answer"])
        self.assertEqual(result["evidence"][0]["quote"], PARENT["passages"][0])

    def test_unverified_collector_draft_is_not_used_as_evidence(self):
        draft = AIMessage(
            content="The deadline is thirty days.", response_metadata={"done": True}
        )
        result = self.run_agent(responses=[draft])
        self.assertNotIn("thirty", result["answer"])
        self.assertIn("seven", result["answer"])

    def test_encoded_claim_arrays_are_not_silently_accepted(self):
        bad = {**ANSWER, "claims": json.dumps(ANSWER["claims"])}
        with self.assertRaises(ValueError):
            self.run_agent(outputs=[bad, bad])

    def test_unanswerable_question_returns_no_evidence(self):
        response = {
            **ANSWER,
            "answerable": False,
            "claims": [],
            "unsupported_parts": ["Tax rate"],
        }
        result = self.run_agent(outputs=[response])
        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["evidence"], [])

    def test_unsupported_parts_prevent_fully_answerable_result(self):
        bad = {**ANSWER, "unsupported_parts": ["Missing condition"]}
        result = self.run_agent(outputs=[bad, ANSWER])
        self.assertEqual(result["coverage"]["unsupported"], [])

    def test_split_identifier_gets_original_reading_context_on_refusal(self):
        missing = {
            **ANSWER,
            "answerable": False,
            "claims": [],
            "unsupported_parts": ["Policy identifier"],
        }
        recovered = {
            **ANSWER,
            "claims": [
                {
                    "text": "Returns are accepted within seven days.",
                    "references": [{"source": 1, "passage": 2}],
                }
            ],
        }
        result = self.run_agent(
            outputs=[missing, recovered], retrieval=FakeRetrieval(preceding=True)
        )
        self.assertIn("seven", result["answer"])
        context = result["parents"][0]["subject_context"][0]["preceding_source"]
        self.assertEqual(context, {"id": "before", "text": "Policy R-17."})
        self.assertEqual(result["parents"][0]["passages"][0], "Policy R-17.")
        self.assertEqual(result["evidence"][0]["quote"], PARENT["text"])
        self.assertEqual(result["steps"][-1]["tool"], "source_context")

    def test_ranking_does_not_reorder_the_reading_passages(self):
        parent = {
            **PARENT,
            "atom_ids": ["a", "b"],
            "passages": [PARENT["text"], "Only unused goods are accepted."],
        }

        class RankedRetrieval(FakeRetrieval):
            def search(self, question):
                return [{"atom_ids": ["b"]}]

            def expand(self, children):
                return [parent]

        retrieval = RankedRetrieval()
        retrieval.graph.add_node("b", text=parent["passages"][1], spans=[], subject="a")
        result = self.run_agent(retrieval=retrieval)
        self.assertEqual(result["parents"][0]["atom_ids"], ["a", "b"])
        self.assertEqual(result["evidence"][0]["quote"], PARENT["text"])

    def test_reading_context_does_not_force_an_unsupported_answer(self):
        missing = {
            **ANSWER,
            "answerable": False,
            "claims": [],
            "unsupported_parts": ["Tax rate"],
        }
        result = self.run_agent(
            outputs=[missing, missing], retrieval=FakeRetrieval(preceding=True)
        )
        self.assertEqual(result["answer"], REFUSAL)
        self.assertEqual(result["evidence"], [])

    def test_writer_retains_history_as_context_not_evidence(self):
        history = [{"role": "user", "content": "We are discussing store returns."}]
        with (
            patch("legal_rag.llm.chat", return_value=json.dumps(ANSWER)) as chat,
            patch("legal_rag.llm.ensure_context", return_value=0),
        ):
            research(
                "And the deadline?",
                cast(Retrieval, FakeRetrieval()),
                Settings(),
                history=history,
                model=ToolModel(responses=[ready()]),
            )
        prompt = chat.call_args.args[0]["messages"][1]["content"]
        self.assertIn(history[0]["content"], prompt)
        self.assertIn("context only, not evidence", prompt)
        supported, unsupported = chat.call_args.args[0]["format"]["oneOf"]
        self.assertTrue(supported["properties"]["answerable"]["const"])
        self.assertEqual(supported["properties"]["claims"]["minItems"], 1)
        self.assertFalse(unsupported["properties"]["answerable"]["const"])
        self.assertEqual(unsupported["properties"]["claims"]["maxItems"], 0)
        schema = chat.call_args.args[0]["format"]
        self.assertEqual(
            schema["$defs"]["Claim"]["properties"]["text"]["maxLength"], 2000
        )
        self.assertEqual(
            schema["$defs"]["SourceReference"]["properties"]["source"]["maximum"], 1000
        )

    def test_incomplete_inference_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            self.run_agent(responses=[ready(done=False)])

    def test_budget_failure_is_not_claimed_as_missing_source(self):
        result = self.run_agent(
            responses=[call("read_section", {"section_id": "p1"})],
            settings=Settings(model_rounds=2),
        )
        self.assertEqual(result["completion"], "model_round_budget_exhausted")
        self.assertEqual(result["coverage"]["unsupported"], [])
        self.assertIn("could not be produced", result["answer"])

    def test_optional_web_access_rejects_private_addresses_and_bad_ports(self):
        with patch(
            "legal_rag.agent.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("127.0.0.1", 443))],
        ):
            self.assertFalse(allowed_url("https://example.gov.vn/page", ("gov.vn",)))
        with patch(
            "legal_rag.agent.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("8.8.8.8", 443))],
        ):
            self.assertTrue(allowed_url("https://example.gov.vn/page", ("gov.vn",)))
        self.assertFalse(
            allowed_url("https://example.gov.vn:invalid/page", ("gov.vn",))
        )
        self.assertFalse(
            allowed_url("https://example.gov.vn.evil.test/page", ("gov.vn",))
        )


if __name__ == "__main__":
    unittest.main()
