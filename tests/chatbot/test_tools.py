import unittest
from unittest.mock import patch

from graphrag.chatbot.tools import search_documents


class SearchDocumentsTests(unittest.TestCase):
    def setUp(self):
        embedding_patch = patch("graphrag.chatbot.tools.generate_embedding")
        driver_patch = patch("graphrag.chatbot.tools.create_neo4j_driver")
        settings_patch = patch("graphrag.chatbot.tools.settings")
        self.embed = embedding_patch.start()
        self.driver_factory = driver_patch.start()
        settings_patch.start()
        self.addCleanup(embedding_patch.stop)
        self.addCleanup(driver_patch.stop)
        self.addCleanup(settings_patch.stop)
        self.driver = self.driver_factory.return_value.__enter__.return_value
        self.session = self.driver.session.return_value.__enter__.return_value
        self.embed.return_value = [0.1, 0.2]
        self.session.run.return_value.data.return_value = [{
            "chunk_id": "roles_001", "source_document": "role_library.pdf",
            "pages": [1, 2], "text": "A role requires Python.",
            "figures_json": "[]", "score": 0.91,
        }]

    def test_returns_vector_results_with_citation_metadata(self):
        results = search_documents("Which role requires Python?")
        self.embed.assert_called_once_with("Which role requires Python?")
        query = self.session.run.call_args.args[0]
        parameters = self.session.run.call_args.kwargs
        self.assertIn("db.index.vector.queryNodes", query)
        self.assertEqual(parameters["top_k"], 5)
        self.assertEqual(parameters["embedding"], [0.1, 0.2])
        self.assertEqual(results[0]["chunk_id"], "roles_001")
        self.assertEqual(results[0]["source_document"], "role_library.pdf")
        self.assertEqual(results[0]["pages"], [1, 2])
        self.assertEqual(results[0]["figures"], [])

    def test_custom_top_k_is_forwarded(self):
        search_documents("Question", top_k=3)
        self.assertEqual(self.session.run.call_args.kwargs["top_k"], 3)

    def test_invalid_inputs_do_not_call_services(self):
        for question, top_k in (("", 5), ("Question", 0), ("Question", 101), ("Question", True)):
            with self.subTest(question=question, top_k=top_k), self.assertRaises(ValueError):
                search_documents(question, top_k)
        self.embed.assert_not_called()
        self.driver_factory.assert_not_called()

    def test_empty_vector_is_rejected_before_database_access(self):
        self.embed.return_value = []
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            search_documents("Question")
        self.driver_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()