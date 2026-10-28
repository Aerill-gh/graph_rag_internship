import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from graphrag.chatbot.agent import AgentAnswer, BaselineAgent, GraphRAGAgent, NOT_FOUND
from graphrag.config import FoundrySettings


def make_response(text="", calls=(), status="completed"):
    return SimpleNamespace(status=status, output_text=text, output=list(calls))


def make_search_call(question="Python", call_id="call-1"):
    return SimpleNamespace(
        type="function_call",
        name="search_documents",
        call_id=call_id,
        arguments=json.dumps({"question": question}),
    )


class BaselineAgentTests(unittest.TestCase):
    def test_graph_agent_registers_three_tools_and_cites_graph_passages(self):
        graph_call = make_search_call()
        graph_call.name = "explore_entity"
        graph_call.arguments = json.dumps({"name": "PerksPlus", "label": "Plan", "limit": 20})
        self.client.responses.create.side_effect = [
            make_response(calls=[graph_call]), make_response("The plan covers fitness [S1]."),
        ]
        with patch("graphrag.chatbot.agent.run_graph_tool", return_value={"passages": self.search.return_value}):
            result = GraphRAGAgent(self.configuration).ask("What does PerksPlus cover?")
        self.assertEqual(result.tools_used, ["explore_entity"])
        definition = self.project.agents.create_version.call_args.kwargs["definition"]
        self.assertEqual({tool.name for tool in definition.tools}, {"search_documents", "explore_entity", "query_graph"})
        self.assertEqual(self.client.responses.create.call_args_list[0].kwargs["tool_choice"], "required")
        self.search.assert_not_called()

    def test_empty_graph_lookup_can_fall_back_to_vector_search(self):
        graph_call = make_search_call()
        graph_call.name = "explore_entity"
        graph_call.arguments = json.dumps({"name": "Missing", "label": "Plan", "limit": 20})
        self.client.responses.create.side_effect = [
            make_response(calls=[graph_call]), make_response(calls=[make_search_call()]),
            make_response("An answer [S1]."),
        ]
        with patch("graphrag.chatbot.agent.run_graph_tool", return_value={"passages": []}):
            result = GraphRAGAgent(self.configuration).ask("Question")
        self.assertEqual(result.tools_used, ["explore_entity", "search_documents"])

    def test_baseline_does_not_register_graph_tools(self):
        self.assertEqual([tool.name for tool in self.agent.tool_definitions()], ["search_documents"])
        with self.assertRaises(ValueError):
            self.agent.run_tool("query_graph", {})

    def setUp(self):
        project_patch = patch("graphrag.chatbot.agent.AIProjectClient")
        credential_patch = patch("graphrag.chatbot.agent.DefaultAzureCredential")
        search_patch = patch("graphrag.chatbot.agent.search_documents")
        self.project_class = project_patch.start()
        credential_patch.start()
        self.search = search_patch.start()
        self.addCleanup(project_patch.stop)
        self.addCleanup(credential_patch.stop)
        self.addCleanup(search_patch.stop)

        self.project = self.project_class.return_value.__enter__.return_value
        self.project.agents.create_version.return_value = SimpleNamespace(
            name="test-agent", version="1"
        )
        self.client = self.project.get_openai_client.return_value.__enter__.return_value
        self.client.conversations.create.return_value = SimpleNamespace(id="conversation-1")
        self.search.return_value = [{
            "chunk_id": "chunk-1", "source_document": "role_library.pdf",
            "pages": [2], "text": "Engineer requires Python.", "score": 0.9,
            "figures": [],
        }]
        self.configuration = FoundrySettings(
            "https://example.services.ai.azure.com/api/projects/demo", "chat-deployment"
        )
        self.agent = BaselineAgent(self.configuration)

    def test_search_evidence_is_returned_with_valid_citation_and_cleanup(self):
        self.client.responses.create.side_effect = [
            make_response(calls=[make_search_call()]),
            make_response("The engineer role requires Python [S1]."),
        ]
        result = self.agent.ask("Which role requires Python?")
        self.assertEqual(result.answer, "The engineer role requires Python [S1].")
        self.assertEqual(result.sources[0]["citation_id"], "S1")
        self.assertEqual(result.sources[0]["pages"], [2])
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.tools_used, ["search_documents"])
        self.assertEqual(result.tool_trace[0]["passages"], 1)
        first = self.client.responses.create.call_args_list[0].kwargs
        self.assertEqual(first["tool_choice"]["name"], "search_documents")
        self.project.agents.delete_version.assert_called_once_with(
            agent_name="test-agent", agent_version="1"
        )
        self.client.conversations.delete.assert_called_once_with(
            conversation_id="conversation-1"
        )

    def test_empty_search_abstains(self):
        self.search.return_value = []
        self.client.responses.create.return_value = make_response(
            calls=[make_search_call()]
        )
        result = self.agent.ask("Unknown topic")
        self.assertEqual(result.answer, NOT_FOUND)
        self.assertEqual(result.sources, [])

    def test_invalid_or_unknown_citations_are_rejected(self):
        for answer in ("Answer without a source.", "Answer [S9]."):
            self.client.responses.create.side_effect = [
                make_response(calls=[make_search_call()]), make_response(answer),
            ]
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                self.agent.ask("Question")

    def test_tool_call_budget_is_enforced(self):
        self.agent.max_tool_calls = 1
        self.client.responses.create.side_effect = [
            make_response(calls=[make_search_call()]),
            make_response(calls=[make_search_call(call_id="call-2")]),
        ]
        with self.assertRaisesRegex(RuntimeError, "budget"):
            self.agent.ask("Question")
        self.assertEqual(self.search.call_count, 1)

    def test_incomplete_response_is_rejected_and_resources_are_cleaned(self):
        self.client.responses.create.return_value = make_response(
            "Partial", status="incomplete"
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.agent.ask("Question")
        self.project.agents.delete_version.assert_called_once()
        self.client.conversations.delete.assert_called_once()

    def test_blank_question_does_not_create_agent(self):
        with self.assertRaises(ValueError):
            self.agent.ask(" ")
        self.project.agents.create_version.assert_not_called()


if __name__ == "__main__":
    unittest.main()