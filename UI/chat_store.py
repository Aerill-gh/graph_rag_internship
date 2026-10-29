import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Iterator, Literal, NotRequired, TypedDict
from uuid import uuid4


DEFAULT_CHAT_PATH = Path(__file__).resolve().parents[1] / "data" / "chats" / "history.sqlite3"
MAX_PINNED_CHATS = 3
MAX_TITLE_LENGTH = 60


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    # Which agent wrote an assistant answer ("baseline" or "graphrag"). Missing for
    # user messages and for answers saved before the mode switch existed.
    mode: NotRequired[str]


@dataclass(frozen=True)
class SavedChat:
    id: str
    title: str
    pinned: bool


def automatic_title(question: str) -> str:
    title = " ".join(question.split())
    if len(title) > MAX_TITLE_LENGTH:
        return title[:MAX_TITLE_LENGTH - 3].rstrip() + "..."
    return title


class ChatStore:
    """Local chat library for this single-user learning app."""

    def __init__(self, path: Path = DEFAULT_CHAT_PATH) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    auto_title INTEGER NOT NULL DEFAULT 1,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL
                );
            """)
            # Migration: databases created before the Baseline/GraphRAG switch lack this column.
            columns = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
            if "mode" not in columns:
                db.execute("ALTER TABLE messages ADD COLUMN mode TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    def list_chats(self) -> list[SavedChat]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, title, pinned FROM chats ORDER BY pinned DESC, updated_at DESC, id"
            ).fetchall()
        return [SavedChat(id, title, bool(pinned)) for id, title, pinned in rows]

    def create_chat(self) -> str:
        chat_id = str(uuid4())
        with self._connect() as db:
            db.execute(
                "INSERT INTO chats (id, title, updated_at) VALUES (?, ?, ?)",
                (chat_id, "New chat", time_ns()),
            )
        return chat_id

    def rename_chat(self, chat_id: str, title: str) -> None:
        title = " ".join(title.split())
        if not title or len(title) > MAX_TITLE_LENGTH:
            raise ValueError(f"Chat names must contain 1 to {MAX_TITLE_LENGTH} characters.")
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE chats SET title = ?, auto_title = 0 WHERE id = ?", (title, chat_id)
            )
            if not cursor.rowcount:
                raise ValueError("This chat no longer exists.")

    def set_pinned(self, chat_id: str, pinned: bool) -> None:
        with self._connect() as db:
            # Serialize the count and update so two tabs cannot exceed the limit.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT pinned FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if row is None:
                raise ValueError("This chat no longer exists.")
            if pinned and not row[0]:
                count = db.execute("SELECT COUNT(*) FROM chats WHERE pinned = 1").fetchone()[0]
                if count >= MAX_PINNED_CHATS:
                    raise ValueError("You can pin at most three chats. Unpin one first.")
            db.execute("UPDATE chats SET pinned = ? WHERE id = ?", (int(pinned), chat_id))

    def delete_chat(self, chat_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def messages(self, chat_id: str) -> list[ChatMessage]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT role, content, mode FROM messages WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        messages = []
        for role, content, mode in rows:
            message = ChatMessage(role=role, content=content)
            if mode:
                message["mode"] = mode
            messages.append(message)
        return messages

    def add_message(
        self,
        chat_id: str,
        role: Literal["user", "assistant"],
        content: str,
        mode: str | None = None,
    ) -> None:
        if role not in ("user", "assistant") or not content.strip():
            raise ValueError("A message needs a user/assistant role and nonblank content.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT auto_title FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if row is None:
                raise ValueError("This chat no longer exists.")
            db.execute(
                "INSERT INTO messages (chat_id, role, content, mode) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, mode),
            )
            if role == "user" and row[0]:
                db.execute(
                    "UPDATE chats SET title = ?, auto_title = 0 WHERE id = ?",
                    (automatic_title(content), chat_id),
                )
            db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (time_ns(), chat_id))
