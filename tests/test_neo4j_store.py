import json
import unittest
from unittest.mock import Mock, patch

from graphrag.neo4j_store import (
    EMBEDDING_DIMENSIONS,
    VECTOR_INDEX_NAME,
    create_neo4j_schema,
    store_chunk,
)


class Neo4jStoreTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        settings_patch = patch("graphrag.neo4j_store.settings")
        self.settings = settings_patch.start()
        self.settings.azure_openai_embedding_deployment = "embedding-deployment"
        self.addCleanup(settings_patch.stop)

    def test_schema_creates_constraints_cosine_index_and_waits(self):
        create_neo4j_schema(self.session)
        calls = self.session.run.call_args_list
        self.assertEqual(len(calls), 4)
        self.assertIn("REQUIRE document.name IS UNIQUE", calls[0].args[0])
        self.assertIn("REQUIRE chunk.chunk_id IS UNIQUE", calls[1].args[0])
        self.assertIn("CREATE VECTOR INDEX chunk_embedding_index", calls[2].args[0])
        self.assertIn(f"vector.dimensions`: {EMBEDDING_DIMENSIONS}", calls[2].args[0])
        self.assertIn("vector.similarity_function`: 'cosine'", calls[2].args[0])
        self.assertTrue(calls[2].args[0].endswith("}}"))
        self.assertIn("db.awaitIndex", calls[3].args[0])
        self.assertEqual(calls[3].kwargs["index_name"], VECTOR_INDEX_NAME)

    def test_invalid_dimensions_are_rejected_without_database_calls(self):
        for dimensions in (0, -1, True, 1.5):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                create_neo4j_schema(self.session, dimensions)
        self.session.run.assert_not_called()

    def test_chunk_storage_links_source_document_and_keeps_figures(self):
        chunk = {
            "chunk_id": "roles_001", "source_document": "role_library.pdf",
            "text": "Engineer requires Python.", "pages": [1, 2],
            "token_count": 4, "figures": [{"image_path": "figure.png"}],
        }
        vector = [0.1] * EMBEDDING_DIMENSIONS
        store_chunk(self.session, chunk, vector)
        query = self.session.run.call_args.args[0]
        parameters = self.session.run.call_args.kwargs
        self.assertIn("MERGE (d:Document", query)
        self.assertIn("MERGE (d)-[:HAS_CHUNK]->(c)", query)
        self.assertEqual(json.loads(parameters["figures_json"]), chunk["figures"])
        self.assertEqual(parameters["embedding"], vector)

    def test_bad_chunk_or_vector_is_rejected_before_database_calls(self):
        chunk = {
            "chunk_id": "roles_001", "source_document": "role_library.pdf",
            "text": "Text", "pages": [1], "token_count": 1,
        }
        with self.assertRaises(ValueError):
            store_chunk(self.session, chunk, [0.1])
        with self.assertRaises(ValueError):
            store_chunk(self.session, {**chunk, "text": " "}, [0.1] * EMBEDDING_DIMENSIONS)
        self.session.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()