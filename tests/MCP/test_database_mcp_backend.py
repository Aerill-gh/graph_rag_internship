import unittest
from unittest.mock import Mock, patch

from database_mcp.config import DatabaseSettings
from database_mcp.contracts import ChunkWrite, ExtractionWrite
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


if __name__ == "__main__":
    unittest.main()
