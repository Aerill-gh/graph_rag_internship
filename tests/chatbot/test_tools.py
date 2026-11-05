import inspect
import unittest
from unittest.mock import patch

from graphrag import tools as chatbot_tools
from graphrag.tools import run_graph_tool, search_documents
from graphrag.database_mcp.contracts import (
    EntityCandidate,
    ExploreEntityResponse,
    FindEntitiesResponse,
    GraphFact,
    QueryGraphResponse,
    SearchDocumentsResponse,
    SearchPassage,
)


class SearchDocumentsTests(unittest.TestCase):
    def setUp(self):
        embedding_patch = patch("graphrag.tools.generate_embedding")
        client_patch = patch("graphrag.tools.DatabaseMcpClient")
        self.embed = embedding_patch.start()
        self.client_factory = client_patch.start()
        self.addCleanup(embedding_patch.stop)
        self.addCleanup(client_patch.stop)
        self.client = self.client_factory.return_value.__enter__.return_value
        self.embed.return_value = [0.1] * 1536

    def test_returns_vector_results_with_citation_metadata(self):
        self.client.search_documents.return_value = SearchDocumentsResponse(passages=[
            SearchPassage(
                chunk_id="roles_001", source_document="role_library.pdf", pages=[1, 2],
                text="A role requires Python.", score=0.91,
            ),
        ])

        results = search_documents("Which role requires Python?")

        self.embed.assert_called_once_with("Which role requires Python?")
        self.client.search_documents.assert_called_once_with([0.1] * 1536, 5)
        self.assertEqual(results[0]["chunk_id"], "roles_001")
        self.assertEqual(results[0]["source_document"], "role_library.pdf")
        self.assertEqual(results[0]["pages"], [1, 2])
        self.assertEqual(results[0]["figures"], [])

    def test_custom_top_k_is_forwarded(self):
        self.client.search_documents.return_value = SearchDocumentsResponse(passages=[])
        search_documents("Question", top_k=3)
        self.client.search_documents.assert_called_once_with([0.1] * 1536, 3)

    def test_invalid_inputs_do_not_call_services(self):
        for question, top_k in (("", 5), ("Question", 0), ("Question", 101), ("Question", True)):
            with self.subTest(question=question, top_k=top_k), self.assertRaises(ValueError):
                search_documents(question, top_k)
        self.embed.assert_not_called()
        self.client_factory.assert_not_called()

    def test_empty_vector_is_rejected_before_database_access(self):
        self.embed.return_value = []
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            search_documents("Question")
        self.client_factory.assert_not_called()

    def test_graph_query_maps_mcp_facts_to_agent_passages(self):
        fact = GraphFact(
            source_labels=["Plan"], source="PerksPlus", relationship="COVERS",
            target_labels=["Service"], target="Gym memberships", chunk_id="one",
            document="PerksPlus.pdf", pages=[3], text="Gym memberships are covered.",
        )
        self.client.query_graph.return_value = QueryGraphResponse(
            facts=[fact], cypher="MATCH ...", parameters={}, result_limit=20,
            truncated=False, scope="Extracted graph facts only.",
        )

        result = run_graph_tool("query_graph", {
            "pattern": "Plan|COVERS|Service", "source_names": ["PerksPlus"],
            "target_name": "", "match_mode": "any", "limit": 20,
        })

        self.assertEqual(result["passages"][0]["source_document"], "PerksPlus.pdf")
        self.assertEqual(result["passages"][0]["graph_fact"]["relationship"], "COVERS")
        self.client.query_graph.assert_called_once_with(
            "Plan|COVERS|Service", source_names=["PerksPlus"], target_name="",
            match_mode="any", limit=20,
        )

    def test_ambiguous_entity_lookup_returns_candidates_not_guessed_facts(self):
        self.client.find_entities.return_value = FindEntitiesResponse(candidates=[
            EntityCandidate(name="Northwind Standard", labels=["Plan"]),
            EntityCandidate(name="Northwind Health Plus", labels=["Plan"]),
        ])

        result = run_graph_tool("explore_entity", {
            "name": "Northwind", "label": "Plan", "limit": 20,
        })

        self.assertTrue(result["ambiguous"])
        self.assertEqual(result["passages"], [])
        self.client.explore_entity.assert_not_called()

    def test_exact_entity_lookup_reads_facts_through_mcp(self):
        self.client.find_entities.return_value = FindEntitiesResponse(candidates=[
            EntityCandidate(name="PerksPlus", labels=["Plan"]),
        ])
        fact = GraphFact(
            source_labels=["Plan"], source="PerksPlus", relationship="COVERS",
            target_labels=["Service"], target="Gym", chunk_id="one",
            document="PerksPlus.pdf", pages=[3], text="Gym memberships are covered.",
        )
        self.client.explore_entity.return_value = ExploreEntityResponse(
            facts=[fact], truncated=False,
        )

        result = run_graph_tool("explore_entity", {
            "name": "PerksPlus", "label": "Plan", "limit": 20,
        })

        self.assertEqual(result["passages"][0]["graph_fact"]["target"], "Gym")
        self.client.explore_entity.assert_called_once_with("PerksPlus", "Plan", 20)

    def test_chatbot_tool_module_does_not_import_neo4j(self):
        source = inspect.getsource(chatbot_tools)
        self.assertNotIn("create_neo4j_driver", source)
        self.assertNotIn("GraphDatabase", source)


if __name__ == "__main__":
    unittest.main()