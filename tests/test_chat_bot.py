import os
import unittest
from unittest.mock import ANY, AsyncMock, Mock, patch
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from starlette.requests import Request

from Chatbot import chat_bot_main
from Chatbot.chat_bot import (
    FoundryChatBot, LangChainChatBot, RAG_SYSTEM_PROMPT, SEARCH_DOCUMENTS_TOOL,
)
from azure.ai.projects.models import PromptAgentDefinition
from openai.types.responses import ResponseFunctionToolCall
from Chatbot.config import ChatSettings, load_chat_settings
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


class TestChatHttpEndpoint(unittest.TestCase):
    def test_chat_only_does_not_open_mcp_connection(self):
        with (
            patch.object(chat_bot_main, "load_chat_settings", return_value=ChatSettings(
                "foundry", "https://example.invalid", "test", "zava",
                use_documents=False,
            )),
            patch.object(chat_bot_main, "FoundryChatBot") as factory,
            patch.object(chat_bot_main, "streamable_http_client") as transport,
        ):
            factory.return_value.chat.return_value = "Hello"
            with TestClient(chat_bot_main.app) as client:
                self.assertEqual(
                    client.post("/chat", json={"message": "Hi"}).json(),
                    {"answer": "Hello"},
                )
                self.assertIsNone(factory.call_args.kwargs["document_search"])
                transport.assert_not_called()
            factory.return_value.close.assert_called_once()

    def test_json_contract_for_both_backends(self):
        for backend in ("foundry", "langchain"):
            with (
                self.subTest(backend=backend),
                patch.object(
                    chat_bot_main,
                    "load_chat_settings",
                    return_value=ChatSettings(
                        backend, "https://example.invalid/api/projects/test", "test", "zava",
                        mcp_server_url="https://example.invalid/mcp",
                    ),
                ),
                patch.object(chat_bot_main, "FoundryChatBot") as foundry,
                patch.object(chat_bot_main, "DefaultAzureCredential"),
                patch.object(chat_bot_main, "get_bearer_token_provider"),
                patch.object(chat_bot_main, "ChatOpenAI") as model,
                patch.object(chat_bot_main, "LangChainChatBot") as langchain,
                patch.object(chat_bot_main, "streamable_http_client") as transport,
                patch.object(chat_bot_main, "ClientSession") as sessions,
            ):
                transport.return_value = AsyncMock()
                transport.return_value.__aenter__.return_value = (Mock(), Mock(), Mock())
                sessions.return_value = AsyncMock()
                model.return_value.root_async_client = AsyncMock()
                factory = foundry if backend == "foundry" else langchain
                factory.return_value.chat.return_value = "Hello"
                with TestClient(chat_bot_main.app) as client:
                    response = client.post("/chat", json={"message": "Hi"})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {"answer": "Hello"})
                    for body in ({"message": ""}, {"message": "   "}, {}):
                        self.assertEqual(client.post("/chat", json=body).status_code, 422)
                    factory.return_value.chat.assert_called_once_with("Hi")


class TestApiBackendSelection(unittest.IsolatedAsyncioTestCase):
    async def test_foundry_remains_default_and_closes(self):
        settings = ChatSettings(
            "foundry", "https://example.invalid", "test", "zava",
            mcp_server_url="https://example.invalid/mcp",
        )
        with (
            patch.object(chat_bot_main, "load_chat_settings", return_value=settings),
            patch.object(chat_bot_main, "FoundryChatBot") as factory,
            patch.object(chat_bot_main, "LangChainChatBot") as langchain_factory,
            patch.object(chat_bot_main, "streamable_http_client") as transport,
            patch.object(chat_bot_main, "ClientSession") as sessions,
        ):
            transport.return_value = AsyncMock()
            transport.return_value.__aenter__.return_value = (Mock(), Mock(), Mock())
            sessions.return_value = AsyncMock()
            factory.return_value.chat.return_value = "Hello"
            async with chat_bot_main.lifespan(chat_bot_main.app):
                request = Request({"type": "http", "app": chat_bot_main.app})
                response = chat_bot_main.chat(
                    chat_bot_main.ChatRequest(message="Hi"), request
                )
                self.assertEqual(response.model_dump(), {"answer": "Hello"})
                factory.assert_called_once_with(
                    endpoint=settings.endpoint, model_name="test", agent_name="zava",
                    document_search=ANY,
                )
                langchain_factory.assert_not_called()
            factory.return_value.close.assert_called_once()
            sessions.return_value.__aenter__.return_value.initialize.assert_awaited_once()
            sessions.return_value.__aexit__.assert_awaited_once()
            transport.return_value.__aexit__.assert_awaited_once()

    async def test_langchain_uses_project_endpoint_and_closes_clients(self):
        settings = ChatSettings(
            "langchain", "https://example.invalid/api/projects/test/", "test", "zava"
        )
        with (
            patch.object(chat_bot_main, "load_chat_settings", return_value=settings),
            patch.object(chat_bot_main, "DefaultAzureCredential") as credential,
            patch.object(chat_bot_main, "get_bearer_token_provider") as provider,
            patch.object(chat_bot_main, "ChatOpenAI") as model,
            patch.object(chat_bot_main, "LangChainChatBot") as factory,
            patch.object(chat_bot_main, "FoundryChatBot") as foundry_factory,
        ):
            async_client = AsyncMock()
            model.return_value.root_async_client = async_client
            async with chat_bot_main.lifespan(chat_bot_main.app):
                self.assertIs(chat_bot_main.app.state.chat_bot, factory.return_value)
                self.assertEqual(
                    model.call_args.kwargs["base_url"],
                    "https://example.invalid/api/projects/test/openai/v1",
                )
                provider.assert_called_once_with(
                    credential.return_value.__enter__.return_value,
                    "https://ai.azure.com/.default",
                )
                foundry_factory.assert_not_called()
            model.return_value.root_client.close.assert_called_once()
            async_client.__aexit__.assert_awaited_once()
            credential.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
