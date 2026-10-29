import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from Chatbot.chat_bot import FoundryChatBot, LangChainChatBot
from UI import streamlit_chat
from UI.chat_store import ChatStore, MAX_TITLE_LENGTH, automatic_title


class TestChatStore(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(directory) / "chats" / "test.sqlite3"
        self.store = ChatStore(self.path)

    def test_title_and_messages_survive_reopening(self):
        chat = self.store.create_chat()
        self.assertEqual(self.store.list_chats()[0].title, "New chat")
        self.store.add_message(chat, "user", "  What are\nour benefits? ")
        self.store.add_message(chat, "assistant", "Example benefits [S1]")
        self.store.add_message(chat, "user", "A follow-up question")
        reopened = ChatStore(self.path)
        self.assertEqual(reopened.list_chats()[0].title, "What are our benefits?")
        self.assertEqual(len(reopened.messages(chat)), 3)
        self.assertEqual(reopened.messages(chat)[1]["content"], "Example benefits [S1]")
        self.assertLessEqual(len(automatic_title("word " * 50)), MAX_TITLE_LENGTH)
        self.assertTrue(automatic_title("word " * 50).endswith("..."))

    def test_rename_is_saved_and_not_replaced_by_first_question(self):
        chat = self.store.create_chat()
        self.store.rename_chat(chat, "  My   benefits  ")
        self.store.add_message(chat, "user", "Question")
        self.assertEqual(ChatStore(self.path).list_chats()[0].title, "My benefits")
        for title in (" \n ", "x" * (MAX_TITLE_LENGTH + 1)):
            with self.assertRaises(ValueError):
                self.store.rename_chat(chat, title)

    def test_delete_removes_only_selected_chat_and_its_messages(self):
        first, second = self.store.create_chat(), self.store.create_chat()
        self.store.add_message(first, "user", "First")
        self.store.add_message(second, "user", "Second")
        self.store.delete_chat(first)
        self.assertEqual(self.store.messages(first), [])
        self.assertEqual([chat.id for chat in self.store.list_chats()], [second])
        self.assertEqual(self.store.messages(second)[0]["content"], "Second")

    def test_pins_are_first_persisted_and_limited_to_exactly_three(self):
        chats = [self.store.create_chat() for _ in range(4)]
        for chat in chats[:3]:
            self.store.set_pinned(chat, True)
        with self.assertRaisesRegex(ValueError, "at most three"):
            self.store.set_pinned(chats[3], True)
        self.store.set_pinned(chats[0], True)
        ordered = ChatStore(self.path).list_chats()
        self.assertTrue(all(chat.pinned for chat in ordered[:3]))
        self.assertEqual(ordered[-1].id, chats[3])
        self.store.set_pinned(chats[0], False)
        self.store.set_pinned(chats[3], True)
        self.assertEqual(sum(chat.pinned for chat in self.store.list_chats()), 3)
        self.store.delete_chat(chats[3])
        self.store.set_pinned(chats[0], True)
        self.assertEqual(sum(chat.pinned for chat in self.store.list_chats()), 3)

    def test_concurrent_pin_attempts_cannot_exceed_limit(self):
        chats = [self.store.create_chat() for _ in range(4)]
        for chat in chats[:2]:
            self.store.set_pinned(chat, True)

        def pin(chat):
            try:
                ChatStore(self.path).set_pinned(chat, True)
            except ValueError:
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(pin, chats[2:]))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(sum(chat.pinned for chat in self.store.list_chats()), 3)

    def test_invalid_messages_and_missing_chats_are_explicit_errors(self):
        chat = self.store.create_chat()
        with self.assertRaises(ValueError):
            self.store.add_message(chat, "user", " ")
        for action in (
            lambda: self.store.add_message("missing", "user", "Hello"),
            lambda: self.store.rename_chat("missing", "Hello"),
            lambda: self.store.set_pinned("missing", True),
        ):
            with self.assertRaisesRegex(ValueError, "no longer exists"):
                action()


class TestSavedChatMemory(unittest.TestCase):
    def test_ui_can_import_from_its_script_directory(self):
        result = subprocess.run(
            [sys.executable, "-c", "import streamlit_chat; assert callable(streamlit_chat.main)"],
            cwd=Path(streamlit_chat.__file__).parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_client_sends_explicit_history_even_when_empty(self):
        for history in ([], [{"role": "user", "content": "My name is Sam"}]):
            response = io.BytesIO(b'{"answer":"Hello"}')
            with patch.object(streamlit_chat, "urlopen", return_value=response) as send:
                streamlit_chat.chat_with_bot("Hi", history=history)
            self.assertEqual(
                json.loads(send.call_args.args[0].data),
                {"message": "Hi", "mode": "baseline", "history": history},
            )

    def test_old_database_gets_mode_column_and_keeps_messages(self):
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "old.sqlite3"
        with sqlite3.connect(path) as db:
            db.executescript("""
                CREATE TABLE chats (id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    auto_title INTEGER NOT NULL DEFAULT 1, pinned INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL);
                CREATE TABLE messages (id INTEGER PRIMARY KEY, chat_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL);
                INSERT INTO chats VALUES ('old', 'Old chat', 0, 0, 1);
                INSERT INTO messages (chat_id, role, content) VALUES ('old', 'assistant', 'Old answer');
            """)
        db.close()
        store = ChatStore(path)
        self.assertEqual(store.messages("old"), [{"role": "assistant", "content": "Old answer"}])
        store.add_message("old", "assistant", "New answer", mode="graphrag")
        self.assertEqual(store.messages("old")[1]["mode"], "graphrag")

    def test_foundry_uses_only_selected_history_and_preserves_legacy_conversation(self):
        with (
            patch("Chatbot.chat_bot.DefaultAzureCredential"),
            patch("Chatbot.chat_bot.AIProjectClient") as projects,
        ):
            project = projects.return_value.__enter__.return_value
            project.agents.get.return_value = None
            project.agents.create_version.return_value.version = "1"
            client = project.get_openai_client.return_value.__enter__.return_value
            client.conversations.create.side_effect = [
                SimpleNamespace(id="legacy"),
                SimpleNamespace(id="saved-a"),
                SimpleNamespace(id="saved-b"),
                SimpleNamespace(id="saved-a-reopened"),
                SimpleNamespace(id="failed-turn"),
            ]
            client.responses.create.return_value = SimpleNamespace(
                status="completed", output=[], output_text="Hello",
            )
            bot = FoundryChatBot("https://example.invalid", "test")
            self.addCleanup(bot.close)
            bot.chat("Legacy question")
            first = [{"role": "user", "content": "My name is Sam"}]
            bot.chat_with_history("What is my name?", first)
            bot.chat_with_history("Hello", [])
            bot.chat_with_history("What is my name?", first)
            bot.chat("Legacy follow-up")
            calls = client.responses.create.call_args_list
            self.assertEqual(
                [call.kwargs["conversation"] for call in calls],
                ["legacy", "saved-a", "saved-b", "saved-a-reopened", "legacy"],
            )
            self.assertEqual(calls[1].kwargs["input"][0]["content"], "My name is Sam")
            self.assertEqual(len(calls[2].kwargs["input"]), 1)
            self.assertEqual(calls[3].kwargs["input"], calls[1].kwargs["input"])
            client.responses.create.side_effect = RuntimeError("Model unavailable")
            with self.assertRaises(RuntimeError):
                bot.chat_with_history("Failure", [])
            self.assertEqual(bot._conversation_id, "legacy")

    def test_langchain_supplied_history_does_not_change_default_thread(self):
        bot = LangChainChatBot(
            FakeMessagesListChatModel(responses=[AIMessage(content="Hello")]),
        )
        bot.chat("Legacy question")
        history = [{"role": "user", "content": "My name is Sam"}]
        with patch.object(bot._history_agent, "invoke", wraps=bot._history_agent.invoke) as invoke:
            self.assertEqual(bot.chat_with_history("Who am I?", history), "Hello")
            self.assertEqual(invoke.call_args.args[0]["messages"], [
                *history, {"role": "user", "content": "Who am I?"},
            ])
            bot.chat_with_history("Another chat", [])
            self.assertEqual(invoke.call_args.args[0]["messages"], [
                {"role": "user", "content": "Another chat"},
            ])
        state = bot.agent.get_state({"configurable": {"thread_id": bot._thread_id}})
        self.assertEqual(len(state.values["messages"]), 2)


class TestSavedChatUI(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.store = ChatStore(Path(directory) / "chats.sqlite3")
        self.enterContext(patch.object(streamlit_chat, "ChatStore", return_value=self.store))
        self.send = self.enterContext(patch.object(
            streamlit_chat, "chat_with_bot",
            return_value=streamlit_chat.ChatAnswer("Answer", "baseline"),
        ))
        self.app = AppTest.from_string(
            "from UI.streamlit_chat import main\nmain()", default_timeout=10,
        ).run()
        self.assertFalse(self.app.exception)

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def test_sidebar_uses_logo_instead_of_text_heading(self):
        self.assertTrue((streamlit_chat.PROJECT_ROOT / "UI" / "assets" / "zava_logo.png").is_file())
        self.assertEqual(len(self.app.sidebar.get("image")), 1)
        self.assertFalse(any("Zava Assistant" in item.value for item in self.app.sidebar.markdown))

    def test_new_switch_rename_pin_delete_and_reload(self):
        first = self.app.session_state.active_chat_id
        self.app.chat_input[0].set_value("My first question").run()
        self.assertFalse(self.app.exception)
        self.send.assert_called_once_with("My first question", history=[], mode="baseline")
        self.assertEqual(self.store.list_chats()[0].title, "My first question")
        self.button("New chat").click().run()
        second = self.app.session_state.active_chat_id
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.app.chat_message), 0)
        self.app.chat_input[0].set_value("Second question").run()
        self.send.assert_called_with("Second question", history=[], mode="baseline")
        self.app.button(key=f"select_{first}").click().run()
        self.assertEqual(len(self.app.chat_message), 2)
        self.app.chat_input[0].set_value("Follow up").run()
        self.send.assert_called_with("Follow up", history=[
            {"role": "user", "content": "My first question"},
            {"role": "assistant", "content": "Answer"},
        ], mode="baseline")
        self.app.button(key=f"rename_{first}").click().run()
        name = next(item for item in self.app.text_input if item.value == "My first question")
        name.set_value("Benefits")
        self.button("Save").click().run()
        self.assertEqual(self.store.list_chats()[0].title, "Benefits")
        self.app.button(key=f"pin_{first}").click().run()
        self.assertTrue(self.store.list_chats()[0].pinned)
        reloaded = AppTest.from_string("from UI.streamlit_chat import main\nmain()").run()
        self.assertFalse(reloaded.exception)
        self.assertEqual(reloaded.session_state.active_chat_id, first)
        self.assertEqual(len(reloaded.chat_message), 4)
        self.app.button(key=f"delete_{first}").click().run()
        self.app.button(key=f"confirm_delete_{first}").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state.active_chat_id, second)
        self.assertEqual(self.store.messages(first), [])

    def test_pin_limit_disables_fourth_pin_and_allows_unpin(self):
        for _ in range(3):
            self.store.create_chat()
        chats = self.store.list_chats()
        for chat in chats[:3]:
            self.store.set_pinned(chat.id, True)
        self.app.run()
        self.assertTrue(self.app.button(key=f"pin_{chats[3].id}").disabled)
        self.app.button(key=f"pin_{chats[0].id}").click().run()
        self.assertFalse(self.app.button(key=f"pin_{chats[3].id}").disabled)

    def test_deleting_last_chat_creates_an_empty_replacement(self):
        old_id = self.app.session_state.active_chat_id
        self.app.button(key=f"delete_{old_id}").click().run()
        self.app.button(key=f"confirm_delete_{old_id}").click().run()
        self.assertFalse(self.app.exception)
        self.assertNotEqual(self.app.session_state.active_chat_id, old_id)
        self.assertEqual(len(self.store.list_chats()), 1)
        self.assertEqual(len(self.app.chat_message), 0)

    def test_chat_menu_has_only_three_actions_and_dialogs_can_be_cancelled(self):
        chat_id = self.app.session_state.active_chat_id
        popovers = self.app.get("popover")
        self.assertEqual(len(popovers), 1)
        self.assertEqual([button.label for button in popovers[0].button], [
            "Pin", "Rename", "Delete",
        ])
        self.assertEqual(len(self.app.text_input), 0)
        self.assertEqual(len(self.app.checkbox), 0)
        self.app.button(key=f"rename_{chat_id}").click().run()
        self.app.button(key=f"cancel_rename_{chat_id}").click().run()
        self.assertEqual(self.store.list_chats()[0].title, "New chat")
        self.app.button(key=f"delete_{chat_id}").click().run()
        self.assertEqual(len(self.store.list_chats()), 1)
        self.app.button(key=f"cancel_delete_{chat_id}").click().run()
        self.assertEqual(self.store.list_chats()[0].id, chat_id)
        self.assertFalse(self.app.exception)

    def test_mode_switch_sends_graphrag_and_labels_each_answer(self):
        self.send.return_value = streamlit_chat.ChatAnswer(
            "Graph answer [S1]", "graphrag",
            sources=[{"citation_id": "S1", "source_document": "PerksPlus.pdf",
                      "chunk_id": "PerksPlus_001", "pages": [1]}],
            tools_used=["explore_entity"],
        )
        self.app.radio(key="agent_mode").set_value("graphrag").run()
        self.app.chat_input[0].set_value("What does PerksPlus cover?").run()
        self.assertFalse(self.app.exception)
        self.send.assert_called_once_with("What does PerksPlus cover?", history=[], mode="graphrag")
        saved = self.store.messages(self.app.session_state.active_chat_id)[1]
        self.assertEqual(saved["mode"], "graphrag")
        self.assertIn("Tools used: explore_entity", saved["content"])
        self.assertIn("GraphRAG", [caption.value.split()[-1] for caption in self.app.caption])

    def test_api_failure_keeps_question_without_inventing_answer(self):
        self.send.side_effect = streamlit_chat.ChatServiceError("Cannot reach chatbot")
        self.app.chat_input[0].set_value("Question while offline").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.error[0].value, "Cannot reach chatbot")
        self.assertEqual(self.store.messages(self.app.session_state.active_chat_id), [
            {"role": "user", "content": "Question while offline"},
        ])

    def test_storage_failure_is_visible(self):
        with (
            patch.object(streamlit_chat, "ChatStore", side_effect=sqlite3.OperationalError("locked")),
            self.assertLogs("UI.streamlit_chat", level="ERROR"),
        ):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertIn("Cannot read or save local chats", self.app.error[0].value)


if __name__ == "__main__":
    unittest.main()
