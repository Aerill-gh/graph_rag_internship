import os
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from Chatbot.chat_bot import (
    FoundryChatBot, LangChainChatBot, RAG_SYSTEM_PROMPT, SEARCH_DOCUMENTS_TOOL,
)
from azure.ai.projects.models import PromptAgentDefinition
from openai.types.responses import ResponseFunctionToolCall
from Chatbot.config import load_chat_settings
from Chatbot.document_tools import DocumentPassage


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, _tools, **_kwargs):
        return self


class TestLangChainChatBot(unittest.TestCase):
    def test_answers_and_remembers_follow_up(self):
        bot = LangChainChatBot(
            FakeMessagesListChatModel(
                responses=[AIMessage(content="Hello, Sam"), AIMessage(content="Sam")]
            )
        )
        self.assertEqual(bot.chat("My name is Sam"), "Hello, Sam")
        self.assertEqual(bot.chat("What is my name?"), "Sam")
        state = bot.agent.get_state(
            {"configurable": {"thread_id": bot._thread_id}}
        )
        self.assertEqual(
            [message.content for message in state.values["messages"]],
            ["My name is Sam", "Hello, Sam", "What is my name?", "Sam"],
        )

    def test_shared_checkpointer_keeps_instances_separate(self):
        memory = InMemorySaver()
        model = FakeMessagesListChatModel(responses=[AIMessage(content="Hello")])
        first = LangChainChatBot(model, checkpointer=memory)
        second = LangChainChatBot(model, checkpointer=memory)
        first.chat("First user")
        second.chat("Second user")
        self.assertNotEqual(first._thread_id, second._thread_id)
        state = second.agent.get_state(
            {"configurable": {"thread_id": second._thread_id}}
        )
        self.assertEqual(len(state.values["messages"]), 2)
        self.assertEqual(state.values["messages"][0].content, "Second user")

    def test_blank_input_does_not_invoke_agent(self):
        bot = LangChainChatBot(
            FakeMessagesListChatModel(responses=[AIMessage(content="Hello")])
        )
        with patch.object(bot.agent, "invoke") as invoke:
            for message in ("", " \n "):
                with self.assertRaises(ValueError):
                    bot.chat(message)
            invoke.assert_not_called()

    def test_content_blocks_return_text(self):
        bot = LangChainChatBot(
            FakeMessagesListChatModel(
                responses=[AIMessage(content=[{"type": "text", "text": "Hello"}])]
            )
        )
        self.assertEqual(bot.chat("Hi"), "Hello")

    def test_invalid_or_empty_final_output_is_explicit(self):
        bot = LangChainChatBot(
            FakeMessagesListChatModel(responses=[AIMessage(content="")])
        )
        with self.assertRaisesRegex(RuntimeError, "empty text"):
            bot.chat("Hi")
        for messages in ([], [HumanMessage(content="Not an answer")]):
            with patch.object(bot.agent, "invoke", return_value={"messages": messages}):
                with self.assertRaisesRegex(RuntimeError, "assistant message"):
                    bot.chat("Hi")

    def test_optional_tool_executes_and_returns_answer(self):
        @tool
        def get_example() -> str:
            """Return an example value for this offline test."""
            return "example evidence"

        model = ToolCallingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_example", "args": {}, "id": "call-1"}],
                ),
                AIMessage(content="The example is ready."),
            ]
        )
        bot = LangChainChatBot(model, tools=[get_example])
        self.assertEqual(bot.chat("Get an example"), "The example is ready.")
        state = bot.agent.get_state(
            {"configurable": {"thread_id": bot._thread_id}}
        )
        tool_messages = [
            message for message in state.values["messages"]
            if isinstance(message, ToolMessage)
        ]
        self.assertEqual(tool_messages[0].content, "example evidence")

    def test_repeated_tool_calls_are_bounded(self):
        @tool
        def repeat() -> str:
            """Return a value for the repeated-call test."""
            return "again"

        bot = LangChainChatBot(
            ToolCallingModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "repeat", "args": {}, "id": f"repeat-{index}"}
                        ],
                    )
                    for index in range(20)
                ]
            ),
            tools=[repeat],
        )
        with self.assertRaises(GraphRecursionError):
            bot.chat("Repeat forever")

    def test_agent_errors_propagate(self):
        bot = LangChainChatBot(
            FakeMessagesListChatModel(responses=[AIMessage(content="Hello")])
        )
        with patch.object(bot.agent, "invoke", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                bot.chat("Hi")


class TestFoundryRegression(unittest.TestCase):
    def test_existing_agent_and_conversation_flow(self):
        with (
            patch("Chatbot.chat_bot.DefaultAzureCredential") as credentials,
            patch("Chatbot.chat_bot.AIProjectClient") as projects,
        ):
            project = projects.return_value.__enter__.return_value
            project.agents.get.return_value = SimpleNamespace(
                versions=SimpleNamespace(latest=SimpleNamespace(
                    version="1",
                    definition=PromptAgentDefinition(
                        model="test-model", instructions=RAG_SYSTEM_PROMPT,
                        tools=[SEARCH_DOCUMENTS_TOOL],
                    ),
                ))
            )
            client = project.get_openai_client.return_value.__enter__.return_value
            client.conversations.create.return_value.id = "conversation-1"
            call = SimpleNamespace(
                status="completed", output=[ResponseFunctionToolCall(
                    type="function_call", name="search_documents", call_id="call-1",
                    arguments='{"question":"Example question"}',
                )],
            )
            final = SimpleNamespace(
                status="completed", output=[], output_text="Hello [S1]",
            )
            client.responses.create.side_effect = [call, final, call, final]
            bot = FoundryChatBot(
                "https://example.invalid", "test-model",
                document_search=Mock(return_value=[DocumentPassage(
                    chunk_id="handbook-1", document="handbook.pdf",
                    pages=[4], text="Example passage",
                )]),
            )
            self.assertIn("Hello [S1]", bot.chat("Hi"))
            self.assertIn("Hello [S1]", bot.chat("Follow up"))
            project.agents.create_version.assert_not_called()
            client.conversations.create.assert_called_once()
            self.assertEqual(
                client.responses.create.call_args_list[2].kwargs["input"],
                [{"type": "message", "role": "user", "content": "Follow up"}],
            )
            bot.close()
            project.get_openai_client.return_value.__exit__.assert_called_once()
            projects.return_value.__exit__.assert_called_once()
            credentials.return_value.__exit__.assert_called_once()


class TestChatSettings(unittest.TestCase):
    def test_chat_only_needs_no_mcp_url_and_rejects_invalid_mode(self):
        env = {
            "FOUNDRY_PROJECT_ENDPOINT": "https://example.invalid/project",
            "FOUNDRY_MODEL_NAME": "test-model",
            "CHATBOT_USE_DOCUMENTS": "false",
        }
        with patch("Chatbot.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            self.assertFalse(load_chat_settings().use_documents)
            os.environ["CHATBOT_USE_DOCUMENTS"] = "true"
            with self.assertRaisesRegex(RuntimeError, "MCP_SERVER_URL"):
                load_chat_settings()
            os.environ["CHATBOT_USE_DOCUMENTS"] = "invalid"
            with self.assertRaisesRegex(ValueError, "CHATBOT_USE_DOCUMENTS"):
                load_chat_settings()

    def test_defaults_and_backend_selection(self):
        env = {
            "FOUNDRY_PROJECT_ENDPOINT": "https://example.invalid/project",
            "FOUNDRY_MODEL_NAME": "test-model",
            "MCP_SERVER_URL": "https://example.invalid/mcp",
        }
        with patch("Chatbot.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            self.assertEqual(load_chat_settings().backend, "foundry")
            os.environ["CHATBOT_BACKEND"] = "langchain"
            self.assertEqual(load_chat_settings().backend, "langchain")
            os.environ["CHATBOT_BACKEND"] = "unknown"
            with self.assertRaisesRegex(ValueError, "CHATBOT_BACKEND"):
                load_chat_settings()

    def test_missing_settings_fail_before_client_creation(self):
        with (
            patch("Chatbot.config.load_dotenv"),
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "FOUNDRY_PROJECT_ENDPOINT"):
                load_chat_settings()


if __name__ == "__main__":
    unittest.main()
