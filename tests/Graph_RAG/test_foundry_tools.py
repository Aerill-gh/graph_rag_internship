import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from azure.ai.projects.models import PromptAgentDefinition
from azure.core.exceptions import ResourceNotFoundError
from mcp.types import CallToolResult, TextContent
from openai.types.responses import ResponseFunctionToolCall
from pydantic import ValidationError

from Chatbot.chat_bot import (
    FoundryChatBot, INSUFFICIENT_EVIDENCE_ANSWER, MAX_SEARCH_CALLS,
    NO_EVIDENCE_ANSWER, RAG_SYSTEM_PROMPT, SEARCH_DOCUMENTS_TOOL,
)
from Chatbot.document_tools import DocumentPassage, McpDocumentSearch


PASSAGE = DocumentPassage(
    chunk_id="test-1", document="example.pdf", pages=[4],
    text="Offline test evidence, not a real company policy.",
)


def tool_response(arguments='{"question":"Example question"}', name="search_documents"):
    return SimpleNamespace(status="completed", output=[ResponseFunctionToolCall(
        type="function_call", name=name, arguments=arguments, call_id="call-1",
    )])


def answer_response(text="An example answer [S1]."):
    return SimpleNamespace(status="completed", output=[], output_text=text)


class TestFoundryTools(unittest.TestCase):
    def setUp(self):
        self.credentials = self.enterContext(patch("Chatbot.chat_bot.DefaultAzureCredential"))
        self.projects = self.enterContext(patch("Chatbot.chat_bot.AIProjectClient"))
        self.project = self.projects.return_value.__enter__.return_value
        self.client = self.project.get_openai_client.return_value.__enter__.return_value
        self.client.conversations.create.return_value.id = "conversation-1"
        self.definition = PromptAgentDefinition(
            model="test", instructions=RAG_SYSTEM_PROMPT, tools=[SEARCH_DOCUMENTS_TOOL],
        )
        self.latest = SimpleNamespace(version="1", definition=self.definition)
        self.project.agents.get.return_value = SimpleNamespace(
            versions=SimpleNamespace(latest=self.latest),
        )
        self.project.agents.create_version.return_value.version = "2"
        self.search = Mock(return_value=[PASSAGE])

    def bot(self):
        bot = FoundryChatBot(
            "https://example.invalid/project", "test", document_search=self.search,
        )
        self.addCleanup(bot.close)
        return bot

    def test_forces_search_and_returns_cited_answer(self):
        self.client.responses.create.side_effect = [tool_response(), answer_response()]
        answer = self.bot().chat("Question")
        self.search.assert_called_once_with("Example question")
        first, second = self.client.responses.create.call_args_list
        self.assertEqual(first.kwargs["tool_choice"], {
            "type": "function", "name": "search_documents",
        })
        self.assertEqual(first.kwargs["extra_body"]["agent_reference"]["version"], "1")
        output = second.kwargs["input"][0]
        self.assertEqual(output["call_id"], "call-1")
        self.assertEqual(json.loads(output["output"]), [
            {"citation_id": "S1", **PASSAGE.model_dump()}
        ])
        self.assertIn("example.pdf; pages: 4; chunk: test-1", answer)

    def test_saved_history_still_forces_fresh_document_search(self):
        self.client.responses.create.side_effect = [tool_response(), answer_response()]
        bot = self.bot()
        history = [
            {"role": "user", "content": "Tell me about benefits"},
            {"role": "assistant", "content": "An earlier answer [S9]"},
        ]
        answer = bot.chat_with_history("What about eligibility?", history)
        first = self.client.responses.create.call_args_list[0]
        self.assertEqual(
            [item["content"] for item in first.kwargs["input"]],
            ["Tell me about benefits", "An earlier answer [S9]", "What about eligibility?"],
        )
        self.assertEqual(first.kwargs["tool_choice"], {
            "type": "function", "name": "search_documents",
        })
        self.search.assert_called_once()
        self.assertIn("example.pdf", answer)
        self.assertNotIn("[S9]", answer)
        self.assertIsNone(bot._conversation_id)

    def test_missing_or_stale_agent_gets_updated_version(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                self.project.agents.get.side_effect = (
                    ResourceNotFoundError("missing") if missing else None
                )
                self.latest.definition = PromptAgentDefinition(model="old")
                self.assertEqual(self.bot()._agent_version, "2")
                self.assertEqual(
                    self.project.agents.create_version.call_args.kwargs["definition"].as_dict(),
                    self.definition.as_dict(),
                )
        self.project.agents.update_details.assert_not_called()

    def test_matching_agent_is_reused(self):
        self.bot()
        self.project.agents.create_version.assert_not_called()

    def test_chat_only_answers_without_tools_and_remembers_conversation(self):
        bot = FoundryChatBot("https://example.invalid/project", "test")
        self.addCleanup(bot.close)
        self.client.responses.create.side_effect = [
            answer_response("Hello"), answer_response("A follow-up answer"),
        ]
        self.assertEqual(bot.chat("Hi"), "Hello")
        self.assertEqual(bot.chat("Follow up"), "A follow-up answer")
        self.client.conversations.create.assert_called_once()
        definition = self.project.agents.create_version.call_args.kwargs["definition"]
        self.assertEqual(definition.tools, [])
        for call in self.client.responses.create.call_args_list:
            self.assertEqual(call.kwargs["tool_choice"], "none")
        self.search.assert_not_called()

    def test_chat_only_rejects_empty_answers_and_unexpected_tools(self):
        bot = FoundryChatBot("https://example.invalid/project", "test")
        self.addCleanup(bot.close)
        for response, error in (
            (answer_response(" "), "empty text"),
            (tool_response(), "chat-only mode"),
        ):
            self.client.responses.create.return_value = response
            with self.assertRaisesRegex(RuntimeError, error):
                bot.chat("Hi")
            self.assertIsNone(bot._conversation_id)

    def test_blank_message_and_answer_without_search_are_rejected(self):
        bot = self.bot()
        with self.assertRaises(ValueError):
            bot.chat("   ")
        self.client.responses.create.assert_not_called()
        self.client.responses.create.return_value = answer_response()
        with self.assertRaisesRegex(RuntimeError, "without searching"):
            bot.chat("Question")

    def test_no_passages_and_insufficient_evidence_decline(self):
        bot = self.bot()
        self.search.return_value = []
        self.client.responses.create.side_effect = [tool_response(), answer_response()]
        self.assertEqual(bot.chat("Question"), NO_EVIDENCE_ANSWER)
        self.search.return_value = [PASSAGE]
        self.client.responses.create.side_effect = [
            tool_response(), answer_response(INSUFFICIENT_EVIDENCE_ANSWER),
        ]
        self.assertEqual(bot.chat("Question"), INSUFFICIENT_EVIDENCE_ANSWER)

    def test_missing_or_unknown_citations_are_errors(self):
        bot = self.bot()
        for text in ("", "No citation", "Wrong citation [S99]"):
            self.client.responses.create.side_effect = [
                tool_response(), answer_response(text),
            ]
            with self.assertRaisesRegex(RuntimeError, "citations"):
                bot.chat("Question")

    def test_invalid_arguments_or_unknown_tools_do_not_call_service(self):
        bot = self.bot()
        for arguments, name in (
            ("{}", "search_documents"), ("[]", "search_documents"),
            ("invalid-json", "search_documents"),
            ('{"question":9}', "search_documents"),
            ('{"question":" "}', "search_documents"),
            ('{"question":"hi","other":true}', "search_documents"),
            ('{"question":"hi"}', "unknown_tool"),
        ):
            self.client.responses.create.return_value = tool_response(arguments, name)
            with self.assertRaises(ValueError):
                bot.chat("Question")
        self.search.assert_not_called()

    def test_service_failure_propagates_and_discards_failed_conversation(self):
        bot = self.bot()
        self.client.responses.create.return_value = tool_response()
        self.search.side_effect = RuntimeError("MCP unavailable")
        with self.assertRaisesRegex(RuntimeError, "MCP unavailable"):
            bot.chat("Question")
        self.assertIsNone(bot._conversation_id)

    def test_repeated_search_is_bounded(self):
        self.client.responses.create.return_value = tool_response()
        with self.assertRaisesRegex(RuntimeError, "call limit"):
            self.bot().chat("Question")
        self.assertEqual(self.search.call_count, MAX_SEARCH_CALLS)

    def test_failed_or_incomplete_response_is_an_error(self):
        bot = self.bot()
        for status in ("failed", "incomplete"):
            self.client.responses.create.return_value = SimpleNamespace(
                status=status, error="error", incomplete_details=None,
            )
            with self.assertRaisesRegex(RuntimeError, status):
                bot.chat("Question")

    def test_initialization_failure_closes_clients(self):
        self.project.agents.get.side_effect = RuntimeError("Azure unavailable")
        with self.assertRaisesRegex(RuntimeError, "Azure unavailable"):
            self.bot()
        self.projects.return_value.__exit__.assert_called_once()
        self.credentials.return_value.__exit__.assert_called_once()

class TestMcpContract(unittest.IsolatedAsyncioTestCase):
    async def test_json_and_structured_passages(self):
        session = AsyncMock()
        search = McpDocumentSearch(session, "team_search")
        for result in (
            CallToolResult(content=[TextContent(
                type="text", text=json.dumps([PASSAGE.model_dump()]),
            )]),
            CallToolResult(content=[], structuredContent={
                "passages": [PASSAGE.model_dump()],
            }),
        ):
            session.call_tool.return_value = result
            self.assertEqual(await search.search_documents("Question"), [PASSAGE])
        session.call_tool.assert_awaited_with(
            "team_search", arguments={"question": "Question"},
        )

    async def test_empty_results_are_distinct_from_errors(self):
        session = AsyncMock()
        search = McpDocumentSearch(session)
        session.call_tool.return_value = CallToolResult(
            content=[TextContent(type="text", text="[]")],
        )
        self.assertEqual(await search.search_documents("Question"), [])
        session.call_tool.return_value = CallToolResult(content=[], isError=True)
        with self.assertRaisesRegex(RuntimeError, "reported an error"):
            await search.search_documents("Question")

    async def test_malformed_results_are_rejected(self):
        session = AsyncMock()
        search = McpDocumentSearch(session)
        for result in (
            CallToolResult(content=[]),
            CallToolResult(content=[], structuredContent={"wrong": []}),
            CallToolResult(content=[TextContent(type="text", text="not-json")]),
            CallToolResult(content=[TextContent(type="text", text='[{"text":"no source"}]')]),
        ):
            session.call_tool.return_value = result
            with self.assertRaises((ValueError, ValidationError)):
                await search.search_documents("Question")

    async def test_network_errors_propagate(self):
        session = AsyncMock()
        session.call_tool.side_effect = ConnectionError("MCP unavailable")
        with self.assertRaises(ConnectionError):
            await McpDocumentSearch(session).search_documents("Question")

    async def test_missing_url_is_explicit(self):
        from Chatbot.config import load_chat_settings
        import os
        with (
            patch("Chatbot.config.load_dotenv"),
            patch.dict(os.environ, {
                "FOUNDRY_PROJECT_ENDPOINT": "https://example.invalid",
                "FOUNDRY_MODEL_NAME": "test",
            }, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "MCP_SERVER_URL"):
                load_chat_settings()
