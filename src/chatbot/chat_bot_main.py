import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from chatbot.agent import BaselineAgent, GraphRAGAgent, HistoryMessage
from chatbot.chat_store import ChatStore
from chatbot.languages import DEFAULT_LANGUAGE, LANGUAGE_NAMES, Language
from database_mcp.client import DatabaseMcpError


logger = logging.getLogger(__name__)

# "baseline" = plain RAG (only search_documents); "graphrag" = the same agent plus graph tools.
AgentMode = Literal["baseline", "graphrag"]


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, content: str) -> str:
        if not content.strip():
            raise ValueError("History content cannot be blank.")
        return content


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatHistoryMessage] | None = None
    mode: AgentMode = "baseline"
    language: Language = DEFAULT_LANGUAGE
    conversation_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, message: str) -> str:
        if not message.strip():
            raise ValueError("Message cannot be blank.")
        return message


class Source(BaseModel):
    citation_id: str
    source_document: str
    chunk_id: str
    pages: list[int | str] = []


class ChatResponse(BaseModel):
    answer: str
    mode: AgentMode
    sources: list[Source]
    tools_used: list[str]


class ConversationResponse(BaseModel):
    """Minimal conversation info sent to frontend."""
    id: str
    title: str
    pinned: bool
    language: Language


class ConversationUpdateRequest(BaseModel):
    """Update request for conversation title or pinned status."""
    title: str | None = None
    pinned: bool | None = None
    language: Language | None = None


class ConversationMessagesResponse(BaseModel):
    """Messages for a conversation."""
    messages: list[dict]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Both agents read the same Foundry settings, so a missing .env value stops startup
    # with a clear message instead of failing on the first question.
    app.state.agents = {"baseline": BaselineAgent(), "graphrag": GraphRAGAgent()}
    app.state.chat_store = ChatStore()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    agent = request.app.state.agents[body.mode]
    history: list[HistoryMessage] = [
        HistoryMessage(role=item.role, content=item.content) for item in body.history or []
    ]
    try:
        result = agent.ask(body.message, history=history, language=body.language)
    except DatabaseMcpError as exc:
        logger.error("%s agent database MCP failed (%s).", body.mode, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail=(
                f"The {body.mode} agent is unavailable. Check Azure sign-in (az login), "
                "Foundry settings, and that the database MCP server is running."
            ),
        ) from None
    except (ValueError, RuntimeError) as exc:
        # The agent ran but its answer failed a check (missing citations, tool budget, ...).
        logger.error("%s agent returned an invalid answer (%s).", body.mode, type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail=f"The {body.mode} agent could not produce a valid cited answer. Please try again.",
        ) from None
    except Exception as exc:
        # Azure sign-in, Foundry, or Neo4j problems. Only the error type is logged, never details.
        logger.error("%s agent is unavailable (%s).", body.mode, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail=(
                f"The {body.mode} agent is unavailable. Check Azure sign-in (az login), "
                "Foundry settings, and that the database MCP server is running."
            ),
        ) from None

    response = ChatResponse(
        answer=result.answer,
        mode=body.mode,
        sources=[
            Source(
                citation_id=source["citation_id"],
                source_document=source["source_document"],
                chunk_id=source["chunk_id"],
                pages=source.get("pages") or [],
            )
            for source in result.sources
        ],
        tools_used=result.tools_used,
    )

    # Auto-save the messages if conversation_id is provided
    if body.conversation_id:
        store = request.app.state.chat_store
        # Reconstruct the answer with sources for storage
        answer_text = result.answer
        if result.sources:
            source_lines = [
                f"- [{s['citation_id']}] {s['source_document']}, pages {', '.join(str(p) for p in s.get('pages') or [])}, chunk {s['chunk_id']}"
                for s in result.sources
            ]
            answer_text += "\n\nSources:\n" + "\n".join(source_lines)
        if result.tools_used:
            answer_text += "\n\nTools used: " + " → ".join(result.tools_used)
        
        store.add_message(body.conversation_id, "user", body.message)
        store.add_message(body.conversation_id, "assistant", answer_text, mode=body.mode)

    return response


@app.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(request: Request) -> list[ConversationResponse]:
    """Get all conversations, sorted by most recent first."""
    store = request.app.state.chat_store
    chats = store.list_chats()
    return [
        ConversationResponse(
            id=chat.id, title=chat.title, pinned=chat.pinned, language=chat.language,
        )
        for chat in chats
    ]


@app.post("/conversations", response_model=ConversationResponse)
def create_conversation(request: Request) -> ConversationResponse:
    """Create a new empty conversation."""
    store = request.app.state.chat_store
    chat_id = store.create_chat()
    chat = next(c for c in store.list_chats() if c.id == chat_id)
    return ConversationResponse(
        id=chat.id, title=chat.title, pinned=chat.pinned, language=chat.language,
    )


@app.get("/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse)
def get_conversation_messages(conversation_id: str, request: Request) -> ConversationMessagesResponse:
    """Get all messages for a conversation."""
    store = request.app.state.chat_store
    messages = store.messages(conversation_id)
    return ConversationMessagesResponse(messages=messages)


@app.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: str, body: ConversationUpdateRequest, request: Request
) -> ConversationResponse:
    """Update conversation title or pinned status."""
    store = request.app.state.chat_store
    chat = next((c for c in store.list_chats() if c.id == conversation_id), None)
    if chat is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if body.title is not None:
        store.rename_chat(conversation_id, body.title)
    if body.pinned is not None:
        store.set_pinned(conversation_id, body.pinned)
    if body.language is not None:
        store.set_language(conversation_id, body.language)

    chat = next((c for c in store.list_chats() if c.id == conversation_id), None)
    return ConversationResponse(
        id=chat.id, title=chat.title, pinned=chat.pinned, language=chat.language,
    )


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request) -> dict:
    """Delete a conversation."""
    store = request.app.state.chat_store
    store.delete_chat(conversation_id)
    return {"status": "deleted"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)