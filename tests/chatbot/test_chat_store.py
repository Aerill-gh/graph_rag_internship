import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from chatbot.chat_store import ChatStore


class ChatStoreLanguageTests(unittest.TestCase):
    def test_new_chat_defaults_to_english_and_language_can_be_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStore(Path(directory) / "chats.sqlite3")
            chat_id = store.create_chat()

            self.assertEqual(store.list_chats()[0].language, "en")
            store.set_language(chat_id, "de")
            self.assertEqual(store.list_chats()[0].language, "de")

    def test_invalid_language_and_missing_chat_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStore(Path(directory) / "chats.sqlite3")
            chat_id = store.create_chat()

            with self.assertRaisesRegex(ValueError, "Unsupported"):
                store.set_language(chat_id, "fr")  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "no longer exists"):
                store.set_language("missing", "es")

    def test_existing_database_migrates_without_losing_chat_or_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "chats.sqlite3"
            with closing(sqlite3.connect(database_path)) as db:
                db.executescript("""
                    CREATE TABLE chats (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        auto_title INTEGER NOT NULL DEFAULT 1,
                        pinned INTEGER NOT NULL DEFAULT 0,
                        updated_at INTEGER NOT NULL
                    );
                    CREATE TABLE messages (
                        id INTEGER PRIMARY KEY,
                        chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        mode TEXT
                    );
                    INSERT INTO chats VALUES ('existing', 'Previous chat', 0, 1, 123);
                    INSERT INTO messages (chat_id, role, content, mode)
                    VALUES ('existing', 'user', 'Earlier question', NULL);
                """)

            store = ChatStore(database_path)

            chat = store.list_chats()[0]
            self.assertEqual((chat.id, chat.title, chat.pinned, chat.language), (
                "existing", "Previous chat", True, "en",
            ))
            self.assertEqual(store.messages("existing"), [
                {"role": "user", "content": "Earlier question"},
            ])


if __name__ == "__main__":
    unittest.main()