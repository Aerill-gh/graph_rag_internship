import unittest
from unittest.mock import Mock, patch

from graphrag.graph_tools import explore_entity, find_entities
from graphrag.tools import _search_chunks, search_documents


class GraphToolsTests(unittest.TestCase):
    def test_document_search_uses_the_configured_database(self):
        with patch("graphrag.tools.settings") as configuration, patch(
            "graphrag.tools.generate_embedding", return_value=[0.1, 0.2]
        ), patch("graphrag.tools.create_neo4j_driver") as factory:
            configuration.neo4j.database = "knowledge_graph"
            driver = factory.return_value.__enter__.return_value
            session = driver.session.return_value.__enter__.return_value
            session.execute_read.return_value = []

            self.assertEqual(search_documents("What does PerksPlus cover?"), [])

            driver.session.assert_called_once_with(database="knowledge_graph")
            session.execute_read.assert_called_once_with(_search_chunks, [0.1, 0.2])
            session.execute_write.assert_not_called()

    def test_queries_use_read_transactions_parameters_and_limits(self):
        for function, arguments in ((find_entities, ("Health",)), (explore_entity, ("Health", "Plan"))):
            with self.subTest(function=function.__name__), patch("graphrag.graph_tools.create_neo4j_driver") as driver:
                session = driver.return_value.__enter__.return_value.session.return_value.__enter__.return_value
                transaction = Mock()
                transaction.run.return_value.data.return_value = [{"name": "Health"}]
                session.execute_read.side_effect = lambda callback: callback(transaction)
                self.assertEqual(function(*arguments), [{"name": "Health"}])
                query = transaction.run.call_args.args[0]
                self.assertIn("LIMIT $limit", query)
                self.assertNotIn("Health", query)
                self.assertEqual(transaction.run.call_args.kwargs["name"], "Health")
                session.execute_write.assert_not_called()

    def test_invalid_inputs_are_rejected(self):
        for limit in (0, 51, True, 1.5):
            with self.assertRaises(ValueError):
                find_entities("Health", limit)
        with self.assertRaises(ValueError):
            explore_entity("Health", "Unapproved")