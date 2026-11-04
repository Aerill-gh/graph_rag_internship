import unittest
from pathlib import Path
from unittest.mock import patch

import graphrag.graph_tools as graph_tools
import graphrag.tools as document_tools
from database_mcp.contracts import (
    EntityCandidate,
    ExploreEntityResponse,
    FindEntitiesResponse,
    GraphFact,
    QueryGraphResponse,
    SearchDocumentsResponse,
    SearchPassage,
)
from graphrag.graph_tools import explore_entity, find_entities, query_graph
from graphrag.tools import search_documents


FACT = GraphFact(
    source_labels=["Plan"], source="PerksPlus", relationship="COVERS",
    target_labels=["Service"], target="Gym", chunk_id="chunk-1",
    document="PerksPlus.pdf", pages=[3], text="Gym memberships are covered.",
)


class GraphToolsTests(unittest.TestCase):
    def test_graph_query_forwards_validated_filters_through_mcp(self):
        response = QueryGraphResponse(
            facts=[FACT], cypher="MATCH ... LIMIT $limit",
            parameters={"source_names": ["CEO"]}, result_limit=2,
            truncated=False, scope="Extracted graph facts only.",
        )
        with patch("graphrag.graph_tools.DatabaseMcpClient") as factory:
            client = factory.return_value.__enter__.return_value
            client.query_graph.return_value = response
            name = "CEO' MATCH (n) DETACH DELETE n //"

            result = query_graph("Role", "REQUIRES", "Skill", [name], limit=2)

        self.assertEqual(result["facts"][0]["source"], "PerksPlus")
        client.query_graph.assert_called_once_with(
            "Role|REQUIRES|Skill", source_names=[name], target_name="",
            match_mode="any", limit=2,
        )

    def test_invalid_graph_requests_are_rejected_before_connecting(self):
        with patch("graphrag.graph_tools.DatabaseMcpClient") as factory:
            for arguments in (
                ("Role) DELETE n //", "REQUIRES", "Skill", ["CEO"], "any"),
                ("Role", "REQUIRES", "Skill", ["CEO"], "difference"),
                ("Role", "REQUIRES", "Skill", ["CEO", "ceo"], "all"),
            ):
                source_label, relationship, target_label, names, mode = arguments
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    query_graph(
                        source_label, relationship, target_label, names, match_mode=mode,
                    )
            factory.assert_not_called()

    def test_graph_query_supports_shared_and_difference_modes(self):
        response = QueryGraphResponse(
            facts=[], cypher="MATCH ...", parameters={}, result_limit=20,
            truncated=False, scope="Extracted graph facts only.",
        )
        with patch("graphrag.graph_tools.DatabaseMcpClient") as factory:
            client = factory.return_value.__enter__.return_value
            client.query_graph.return_value = response
            query_graph("Role", "REQUIRES", "Skill", ["CEO", "CFO"], match_mode="all")
            query_graph(
                "Plan", "COVERS", "Service", ["Plus", "Standard"],
                match_mode="difference",
            )
        self.assertEqual(client.query_graph.call_args_list[0].kwargs["match_mode"], "all")
        self.assertEqual(
            client.query_graph.call_args_list[1].kwargs["match_mode"], "difference",
        )

    def test_document_search_uses_mcp_and_preserves_citations(self):
        passage = SearchPassage(
            chunk_id="chunk-1", source_document="PerksPlus.pdf", pages=[3],
            text="Gym memberships are covered.", score=0.9,
        )
        with patch("graphrag.tools.generate_embedding", return_value=[0.1] * 1536), patch(
            "graphrag.tools.DatabaseMcpClient",
        ) as factory:
            client = factory.return_value.__enter__.return_value
            client.search_documents.return_value = SearchDocumentsResponse(passages=[passage])

            results = search_documents("What does PerksPlus cover?")

        self.assertEqual(results[0]["document"], "PerksPlus.pdf")
        self.assertEqual(results[0]["chunk_id"], "chunk-1")
        client.search_documents.assert_called_once_with([0.1] * 1536, 3)

    def test_entity_helpers_map_mcp_responses(self):
        with patch("graphrag.graph_tools.DatabaseMcpClient") as factory:
            client = factory.return_value.__enter__.return_value
            client.find_entities.return_value = FindEntitiesResponse(candidates=[
                EntityCandidate(labels=["Plan"], name="Health"),
            ])
            client.explore_entity.return_value = ExploreEntityResponse(
                facts=[FACT], truncated=False,
            )

            self.assertEqual(find_entities("Health")[0]["name"], "Health")
            self.assertEqual(explore_entity("Health", "Plan")[0]["target"], "Gym")

    def test_invalid_search_inputs_do_not_connect_to_mcp(self):
        with patch("graphrag.tools.generate_embedding") as embedding, patch(
            "graphrag.tools.DatabaseMcpClient",
        ) as factory:
            with self.assertRaises(ValueError):
                search_documents(" ")
            embedding.assert_not_called()
            factory.assert_not_called()

    def test_graph_helpers_do_not_import_neo4j(self):
        for module in (document_tools, graph_tools):
            with self.subTest(module=module.__name__):
                source = Path(module.__file__).read_text(encoding="utf-8")
                self.assertNotIn("create_neo4j_driver", source)
                self.assertNotIn("GraphDatabase", source)