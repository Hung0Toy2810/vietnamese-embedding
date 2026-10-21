"""Meaning-based graph tests: no page, title or font can create semantic edges."""

import copy
import json
import unittest
from unittest.mock import patch

import networkx as nx
import torch
from tokenizers import Tokenizer

from legal_rag.common import token_ranges
from legal_rag.schemas import Atom, ChunkPlan, ParsedDocument, ScopeGraph, Span
from legal_rag.semantic import (
    atomize,
    build_graph,
    candidate_pairs,
    make_chunks,
    relation_batches,
    shortlist_pairs,
    validate_chunks,
    validate_scopes,
)
from legal_rag.settings import Settings
from utils import DEFAULT_TOKENIZER


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
        self.settings = Settings(plan_attempts=1)
        self.plan = ChunkPlan(
            rationale="Keep conditions with their subject", similarity_quantile=0
        )
        self.atoms = [
            Atom(
                id="a",
                text="Customers may return items within seven days.",
                start=0,
                end=44,
                spans=[Span(start=0, end=44, page=1)],
            ),
            Atom(
                id="b",
                text="Disconnect power before cleaning the machine.",
                start=45,
                end=89,
                spans=[Span(start=45, end=89, page=1)],
            ),
            Atom(
                id="c",
                text="Returns do not apply to personalized items.",
                start=90,
                end=132,
                spans=[Span(start=90, end=132, page=4)],
            ),
        ]

    def graph(self, output):
        aliases = {atom.id: f"u{i}" for i, atom in enumerate(self.atoms)}
        subjects = []
        assignments = {atom.id: None for atom in self.atoms}
        kinds = {}
        for edge in output["relations"]:
            anchor = edge["source"]
            if assignments[anchor] is None:
                subjects.append({"anchor": aliases[anchor], "name": "Fixture subject"})
                assignments[anchor] = anchor
            assignments[edge["target"]] = anchor
            kinds[edge["target"]] = edge["kind"]
        scope = ScopeGraph.model_validate(
            {
                "subjects": subjects,
                "members": [
                    {
                        "unit": aliases[unit],
                        "subject": aliases[anchor] if anchor else None,
                        "relation": kinds.get(
                            unit, "introduces" if anchor else "unresolved"
                        ),
                    }
                    for unit, anchor in assignments.items()
                ],
            }
        )

        def response(prompt, schema, settings):
            data = {}
            for name in schema.model_fields:
                data[name] = getattr(scope, name)
            return schema.model_validate(data)

        with patch("legal_rag.semantic.structured", side_effect=response):
            return build_graph(
                self.atoms,
                [(0, 1, 0.8), (0, 2, 0.9)],
                self.plan,
                self.settings,
                lambda _: None,
                lambda _, f: f(),
            )

    def test_same_page_does_not_imply_a_semantic_relation(self):
        graph = self.graph({"relations": []})
        self.assertTrue(graph.has_edge("a", "b"))
        self.assertFalse(
            any(data.get("grouping") for _, _, data in graph.edges(data=True))
        )
        parents, _ = make_chunks(
            graph, self.atoms, self.plan, self.tokenizer, "doc", "file"
        )
        self.assertEqual(len(parents), 3)

    def test_distant_exception_can_share_a_parent(self):
        graph = self.graph(
            {
                "relations": [
                    {
                        "source": "c",
                        "target": "a",
                        "kind": "exception_to",
                        "reason": "Personalized items limit the returns policy.",
                    }
                ]
            }
        )
        parents, children = make_chunks(
            graph, self.atoms, self.plan, self.tokenizer, "doc", "file"
        )
        self.assertTrue(
            any(set(parent["atom_ids"]) == {"a", "c"} for parent in parents)
        )
        self.assertTrue(
            all(
                len(self.tokenizer.encode(child["text"]).ids) <= 256
                for child in children
            )
        )

    def test_unknown_missing_and_conflicting_scope_assignments_are_rejected(self):
        base = {
            "subjects": [{"anchor": "u0", "name": "Returns"}],
            "members": [
                {"unit": "u0", "subject": "u0", "relation": "introduces"},
                {"unit": "u1", "subject": "u0", "relation": "has_condition"},
            ],
        }
        self.assertEqual(
            len(validate_scopes(base, {"u0", "u1"}, {"u0", "u1"}, {}).members), 2
        )
        for changed, known in (
            ({**base, "subjects": [{"anchor": "invented", "name": "Missing"}]}, {}),
            ({**base, "members": base["members"][:1]}, {}),
            (base, {"u1": "other"}),
        ):
            with self.assertRaises(ValueError):
                validate_scopes(changed, {"u0", "u1"}, {"u0", "u1"}, known)

    def test_echoed_context_anchor_is_checked_then_omitted(self):
        output: dict = {
            "subjects": [{"anchor": "u0", "name": "Returns"}],
            "members": [
                {"unit": "u0", "subject": "u0", "relation": "introduces"},
                {"unit": "u1", "subject": "u0", "relation": "has_condition"},
            ],
        }
        result = validate_scopes(output, {"u1"}, {"u0", "u1", "u2"}, {"u1": "u0"})
        self.assertEqual([member.unit for member in result.members], ["u1"])
        output["members"].append(
            {"unit": "u2", "subject": "u0", "relation": "has_condition"}
        )
        with self.assertRaises(ValueError):
            validate_scopes(output, {"u1"}, {"u0", "u1", "u2"}, {})
        output["members"] = output["members"][:2]
        output["members"][0]["subject"] = None
        with self.assertRaises(ValueError):
            validate_scopes(output, {"u1"}, {"u0", "u1"}, {})

    def test_literal_null_subject_is_unresolved_not_an_invented_anchor(self):
        result = validate_scopes(
            {
                "subjects": [],
                "members": [{"unit": "u0", "subject": "null", "relation": "metadata"}],
            },
            {"u0"},
            {"u0"},
            {},
        )
        self.assertIsNone(result.members[0].subject)

    def test_relation_labels_are_not_limited_to_legal_documents(self):
        graph = self.graph(
            {
                "relations": [
                    {
                        "source": "c",
                        "target": "a",
                        "kind": "product_customization_restriction",
                        "reason": "The text restricts product returns.",
                    }
                ]
            }
        )
        self.assertTrue(
            any(
                data.get("kind") == "product_customization_restriction"
                for _, _, data in graph.edges(data=True)
            )
        )

    def test_line_wrapping_does_not_change_language_windows(self):
        raw = "Customers may return items. Disconnect power before cleaning. Personalized items cannot be returned."
        versions = [raw, raw.replace(" ", "\n"), raw.replace(" ", "   ")]
        results = []
        for text in versions:
            parsed = ParsedDocument(
                text=text,
                spans=[Span(start=0, end=len(text), page=1)],
                parser="fixture",
            )
            results.append(
                [
                    " ".join(atom.text.split())
                    for atom in atomize(parsed, "doc", self.tokenizer, 24)
                ]
            )
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_large_sentence_preserves_all_nonwhitespace_content(self):
        text = " ".join(["retrieval"] * 1000)
        parsed = ParsedDocument(
            text=text, spans=[Span(start=0, end=len(text))], parser="fixture"
        )
        atoms = atomize(parsed, "doc", self.tokenizer, 48)
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(atom.id for atom in atoms)
        parents, children = make_chunks(
            graph, atoms, self.plan, self.tokenizer, "doc", "long sentence"
        )
        self.assertEqual(
            "".join(child["text"] for child in children).replace(" ", ""),
            text.replace(" ", ""),
        )
        broken = copy.deepcopy(children)
        broken[0]["text"] = "invented"
        with self.assertRaises(ValueError):
            validate_chunks(parents, broken, atoms, self.tokenizer)

    def test_candidate_search_excludes_self_and_deduplicates_pairs(self):
        pairs = candidate_pairs(torch.eye(4), 2)
        self.assertTrue(all(a < b for a, b, _ in pairs))
        self.assertEqual(len(pairs), len({(a, b) for a, b, _ in pairs}))
        self.assertEqual(candidate_pairs(torch.ones(1, 3), 4), [])

    def test_shortlist_preserves_minority_topic_candidates_without_forcing_edges(self):
        pairs = [(0, 1, 0.95), (0, 2, 0.9), (1, 2, 0.89), (3, 4, 0.3)]
        self.assertIn((3, 4, 0.3), shortlist_pairs(pairs, 0.9))
        restored = json.loads(json.dumps(pairs))
        self.assertIn([3, 4, 0.3], shortlist_pairs(restored, 0.9))
        self.assertEqual(shortlist_pairs([], 0.9), [])
        graph = self.graph({"relations": []})
        self.assertFalse(
            any(data.get("grouping") for _, _, data in graph.edges(data=True))
        )

    def test_language_windows_can_propose_a_dependency_missed_by_embedding(self):
        with patch("legal_rag.semantic.token_count", return_value=100):
            batches = relation_batches(self.atoms, [(0, 1, 0.99)], 0.9, self.settings)
        self.assertTrue(any(0 in batch.nodes and 2 in batch.nodes for batch in batches))

    def test_overlapping_windows_cover_every_unit_without_becoming_edges(self):
        atoms = [
            atom.model_copy(update={"id": str(i)})
            for i, atom in enumerate(self.atoms * 5)
        ]
        with patch("legal_rag.semantic.token_count", return_value=200):
            batches = relation_batches(
                atoms, [], 0.9, Settings(graph_window_tokens=512)
            )
        covered = {node for batch in batches for node in batch.nodes}
        self.assertEqual(covered, set(range(len(atoms))))

    def test_edge_evidence_is_copied_from_original_endpoints(
        self,
    ):
        output = {
            "relations": [
                {
                    "source": "a",
                    "target": "c",
                    "kind": "exception_to",
                    "reason": "A returns restriction",
                }
            ]
        }
        graph = self.graph(output)
        edge = graph["a"]["c"]["membership"]
        self.assertEqual(edge["source_quote"], self.atoms[0].text)
        self.assertEqual(edge["target_quote"], self.atoms[2].text)
        self.assertEqual(edge["verification"], "llm_with_source_id_validation")

    def test_zero_token_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            token_ranges("text", self.tokenizer, 0)

    def test_cross_subject_navigation_does_not_merge_chunks(self):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(atom.id for atom in self.atoms)
        graph.add_edge("a", "c", grouping=True, semantic=True, weight=1.0)
        graph.add_edge("c", "b", grouping=False, semantic=True, weight=1.0)
        parents, _ = make_chunks(
            graph, self.atoms, self.plan, self.tokenizer, "doc", "file"
        )
        self.assertEqual(
            {frozenset(parent["atom_ids"]) for parent in parents},
            {frozenset({"a", "c"}), frozenset({"b"})},
        )

    def test_rejected_distant_candidates_do_not_become_edges(self):
        from legal_rag.semantic import RelationBatch
        from legal_rag.schemas import Relations

        review = Relations.model_validate(
            {
                "relations": [
                    {
                        "source": "u0",
                        "target": "u1",
                        "related": False,
                        "kind": "unrelated",
                        "reason": "Cleaning and returns are different policies.",
                    },
                    {
                        "source": "u2",
                        "target": "u0",
                        "related": True,
                        "kind": "exception_to",
                        "reason": "Customized goods cannot be returned.",
                    },
                ]
            }
        )
        batches = [RelationBatch([0, 1, 2], [(0, 1, 0.8), (0, 2, 0.9), (1, 2, 0.7)])]
        with (
            patch("legal_rag.semantic.relation_batches", return_value=batches),
            patch("legal_rag.semantic.structured", return_value=review),
        ):
            graph = build_graph(
                self.atoms,
                [],
                self.plan,
                self.settings,
                lambda _: None,
                lambda _, compute: compute(),
            )
        edges = [(a, b) for a, b, d in graph.edges(data=True) if d.get("semantic")]
        self.assertEqual(edges, [("c", "a")])
        self.assertEqual(graph.graph["distant_requested_pairs"], 3)
        self.assertEqual(graph.graph["distant_reviewed_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
