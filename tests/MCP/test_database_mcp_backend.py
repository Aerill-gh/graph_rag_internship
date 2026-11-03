import unittest
from unittest.mock import Mock, patch

from database_mcp.config import DatabaseSettings
from database_mcp.contracts import (
    ChunkWrite,
    CorpusSummaryResponse,
    ExploreEntityRequest,
    ExtractionWrite,
    QueryGraphRequest,
    SearchDocumentsRequest,
)
from database_mcp.neo4j_backend import Neo4jBackend


CHUNK = {
    "chunk_id": "test-one", "source_document": "example.pdf", "text": "Test evidence.",
    "pages": [1], "token_count": 2, "figures": [],
}
EXTRACTION = {
    "entities": [{"label": "Role", "name": "Engineer"}], "relationships": [],
}


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = Neo4jBackend(DatabaseSettings(
            uri="bolt://localhost:7687", username="test", password="test-only",
            database="test-database",
        ))
        self.driver = Mock()
        self.session = Mock()
        self.driver.session.return_value = Mock(
            __enter__=Mock(return_value=self.session), __exit__=Mock(return_value=False),
        )
        self.backend.driver = self.driver

    def test_chunk_write_is_managed_and_preserves_vectors_without_embedding(self):
        transaction = Mock()
        self.session.execute_write.side_effect = lambda callback: callback(transaction)
        response = self.backend.upsert_chunk(ChunkWrite.model_validate({"chunk": CHUNK}))
        self.assertEqual(response.chunk_id, CHUNK["chunk_id"])
        self.driver.session.assert_called_once_with(database="test-database")
        self.session.execute_write.assert_called_once()
        parameters = transaction.run.call_args.kwargs
        self.assertIsNone(parameters["embedding"])
        self.assertIsNone(parameters["embedding_model"])
        self.assertIn("MERGE", transaction.run.call_args.args[0])
        self.assertIn("coalesce($embedding, c.embedding)", transaction.run.call_args.args[0])

    def test_missing_source_chunk_rejects_atomic_extraction_write(self):
        transaction = Mock()
        transaction.run.return_value.single.return_value = None
        self.session.execute_write.side_effect = lambda callback: callback(transaction)
        request = ExtractionWrite.model_validate({
            "chunk_id": CHUNK["chunk_id"], "extraction": EXTRACTION,
        })
        with self.assertRaisesRegex(ValueError, "must be stored"):
            self.backend.upsert_extraction(request)
        self.session.execute_write.assert_called_once()
        self.assertEqual(transaction.run.call_count, 1)

    def test_readback_uses_managed_read_and_bounded_fact_queries(self):
        transaction = Mock()
        chunk_result, entities_result, facts_result = Mock(), Mock(), Mock()
        chunk_result.single.return_value = {"chunk": {
            **CHUNK, "figures_json": "[]",
        }}
        entities_result.data.return_value = EXTRACTION["entities"]
        facts_result.data.return_value = []
        transaction.run.side_effect = [chunk_result, entities_result, facts_result]
        self.session.execute_read.side_effect = lambda callback: callback(transaction)
        response = self.backend.get_chunk(CHUNK["chunk_id"])
        self.assertEqual(response.chunk.text, CHUNK["text"])
        self.assertIsNone(response.embedding)
        self.assertEqual(response.entities[0].name, "Engineer")
        self.session.execute_read.assert_called_once()
        for call in transaction.run.call_args_list:
            self.assertEqual(call.kwargs, {"chunk_id": CHUNK["chunk_id"]})
        for call in transaction.run.call_args_list[1:]:
            self.assertIn("LIMIT 51", call.args[0])

    def test_driver_lifecycle_and_connectivity(self):
        self.backend.close()
        self.driver.close.assert_called_once()
        with patch("database_mcp.neo4j_backend.GraphDatabase.driver", return_value=self.driver):
            self.backend.open()
            self.backend.health_check()
        self.driver.verify_connectivity.assert_called_once()

    def test_vector_search_uses_read_transaction_and_returns_passage_metadata(self):
        transaction = Mock()
        transaction.run.return_value.data.return_value = [{
            "chunk_id": "test-one", "source_document": "example.pdf", "pages": [2],
            "text": "Test evidence.", "figures_json": "[]", "score": 0.9,
        }]
        self.session.execute_read.side_effect = lambda callback: callback(transaction)

        response = self.backend.search_documents(SearchDocumentsRequest(
            embedding=[0.1] * 1536, top_k=4,
        ))

        self.assertEqual(response.passages[0].chunk_id, "test-one")
        self.assertEqual(response.passages[0].figures, [])
        self.assertEqual(response.passages[0].score, 0.9)
        query = transaction.run.call_args.args[0]
        self.assertIn("db.index.vector.queryNodes", query)
        self.assertEqual(transaction.run.call_args.kwargs["top_k"], 4)
        self.assertEqual(len(transaction.run.call_args.kwargs["embedding"]), 1536)
        self.session.execute_read.assert_called_once()
        self.session.execute_write.assert_not_called()

    def test_graph_query_is_allowlisted_parameterized_and_bounded(self):
        transaction = Mock()
        fact = {
            "source_labels": ["Plan"], "source": "Plus", "relationship": "COVERS",
            "target_labels": ["Service"], "target": "Gym", "chunk_id": "test-one",
            "document": "example.pdf", "pages": [1], "text": "Gym coverage.",
        }
        transaction.run.return_value.data.return_value = [fact, fact]
        self.session.execute_read.side_effect = lambda callback: callback(transaction)
        hostile_name = "Plus' MATCH (n) DETACH DELETE n //"
        request = QueryGraphRequest(
            pattern="Plan|COVERS|Service", source_names=[hostile_name], limit=1,
        )

        response = self.backend.query_graph(request)

        self.assertEqual(len(response.facts), 1)
        self.assertTrue(response.truncated)
        self.assertNotIn(hostile_name, response.cypher)
        self.assertIn("LIMIT $limit", response.cypher)
        self.assertEqual(transaction.run.call_args.kwargs["source_names"], [hostile_name])
        self.assertEqual(transaction.run.call_args.kwargs["limit"], 2)
        self.session.execute_write.assert_not_called()

    def test_invalid_graph_request_is_rejected_by_contract(self):
        for values in (
            {"pattern": "Role) DELETE n //|REQUIRES|Skill", "source_names": ["CEO"]},
            {"pattern": "Plan|COVERS|Service", "source_names": ["Plus", "plus"],
             "match_mode": "all"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                QueryGraphRequest.model_validate(values)
        self.session.execute_read.assert_not_called()

    def test_explore_entity_uses_a_bounded_read_transaction(self):
        transaction = Mock()
        transaction.run.return_value.data.return_value = [{
            "source_labels": ["Plan"], "source": "Plus", "relationship": "COVERS",
            "target_labels": ["Service"], "target": "Gym", "chunk_id": "test-one",
            "document": "example.pdf", "pages": [1], "text": "Gym coverage.",
        }]
        self.session.execute_read.side_effect = lambda callback: callback(transaction)

        response = self.backend.explore_entity(ExploreEntityRequest(
            name="Plus", label="Plan", limit=10,
        ))

        self.assertEqual(response.facts[0].target, "Gym")
        self.assertEqual(transaction.run.call_args.kwargs["name"], "Plus")
        self.assertIn("LIMIT $limit", transaction.run.call_args.args[0])
        self.session.execute_write.assert_not_called()

    def test_corpus_summary_returns_only_bounded_snapshot_metadata(self):
        transaction = Mock()
        nodes_result, relationships_result = Mock(), Mock()
        nodes_result.data.return_value = [
            {"labels": ["Chunk"], "properties": {"chunk_id": "one", "text": "Evidence"}},
            {"labels": ["Plan"], "properties": {"name": "Plus"}},
        ]
        relationships_result.data.return_value = [{
            "source_labels": ["Plan"], "source": "Plus", "type": "COVERS",
            "target_labels": ["Service"], "target": "Gym", "properties": {"id": 1},
        }]
        transaction.run.side_effect = [nodes_result, relationships_result]
        self.session.execute_read.side_effect = lambda callback: callback(transaction)

        response = self.backend.corpus_summary()

        self.assertIsInstance(response, CorpusSummaryResponse)
        self.assertEqual((response.database, response.chunks, response.nodes, response.relationships), (
            "test-database", 1, 2, 1,
        ))
        self.assertEqual(len(response.sha256), 64)
        for call in transaction.run.call_args_list:
            self.assertIn("LIMIT $limit", call.args[0])
            self.assertEqual(call.kwargs["limit"], 10_001)
        self.session.execute_read.assert_called_once()
        self.session.execute_write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
