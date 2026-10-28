import unittest
from unittest.mock import Mock, patch

from graphrag.graph_tools import explore_entity, find_entities, query_graph
from graphrag.tools import _search_chunks, search_documents


class GraphToolsTests(unittest.TestCase):
    def test_graph_query_is_parameterized_read_only_and_bounded(self):
        with patch("graphrag.graph_tools.create_neo4j_driver") as factory:
            session = factory.return_value.__enter__.return_value.session.return_value.__enter__.return_value
            transaction = Mock()
            transaction.run.return_value.data.return_value = [{"source": "CEO"}] * 3
            session.execute_read.side_effect = lambda callback: callback(transaction)
            name = "CEO' MATCH (n) DETACH DELETE n //"
            result = query_graph("Role", "REQUIRES", "Skill", [name], limit=2)
            self.assertNotIn(name, result["cypher"])
            self.assertIn("LIMIT $limit", result["cypher"])
            self.assertEqual(transaction.run.call_args.kwargs["source_names"], [name])
            self.assertEqual(transaction.run.call_args.kwargs["limit"], 3)
            self.assertEqual(session.execute_read.call_args.args[0].timeout, 10)
            self.assertTrue(result["truncated"])
            self.assertEqual(len(result["facts"]), 2)
            session.execute_write.assert_not_called()

    def test_graph_query_rejects_unapproved_patterns_before_connecting(self):
        with patch("graphrag.graph_tools.create_neo4j_driver") as factory:
            with self.assertRaises(ValueError):
                query_graph("Role) DELETE n //", "REQUIRES", "Skill", ["CEO"])
            with self.assertRaises(ValueError):
                query_graph("Role", "REQUIRES", "Skill", ["CEO"], match_mode="difference")
            with self.assertRaises(ValueError):
                query_graph("Role", "REQUIRES", "Skill", ["CEO", "ceo"])
            factory.assert_not_called()

    def test_graph_query_supports_shared_and_difference_plans(self):
        with patch("graphrag.graph_tools.create_neo4j_driver") as factory:
            session = factory.return_value.__enter__.return_value.session.return_value.__enter__.return_value
            session.execute_read.return_value = []
            shared = query_graph("Role", "REQUIRES", "Skill", ["CEO", "CFO"], match_mode="all")
            difference = query_graph("Plan", "COVERS", "Service", ["Plus", "Standard"], match_mode="difference")
            self.assertIn("all(name IN $source_names", shared["cypher"])
            self.assertIn("NOT EXISTS", difference["cypher"])

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