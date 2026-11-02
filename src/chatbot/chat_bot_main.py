import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from graphrag.chatbot.agent import BaselineAgent, GraphRAGAgent, HistoryMessage


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Both agents read the same Foundry settings, so a missing .env value stops startup
    # with a clear message instead of failing on the first question.
    app.state.agents = {"baseline": BaselineAgent(), "graphrag": GraphRAGAgent()}
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    agent = request.app.state.agents[body.mode]
    history: list[HistoryMessage] = [
        HistoryMessage(role=item.role, content=item.content) for item in body.history or []
    ]
    try:
        result = agent.ask(body.message, history=history)
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
                "Foundry settings, and that Neo4j is running."
            ),
        ) from None

    return ChatResponse(
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)