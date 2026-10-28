from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from typing import AsyncGenerator, Literal

import uvicorn
from anyio import from_thread
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field, field_validator

from Chatbot.chat_bot import FoundryChatBot, HistoryMessage, LangChainChatBot
from Chatbot.config import load_chat_settings
from Chatbot.document_tools import McpDocumentSearch


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

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, message: str) -> str:
        if not message.strip():
            raise ValueError("Message cannot be blank.")
        return message


class ChatResponse(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = load_chat_settings()
    async with AsyncExitStack() as stack:
        if settings.backend == "foundry":
            document_search = None
            if settings.use_documents:
                if not settings.mcp_server_url:
                    raise RuntimeError("MCP_SERVER_URL is required for document-grounded chat.")
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamable_http_client(settings.mcp_server_url)
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                document_tools = McpDocumentSearch(session, settings.mcp_search_tool)
                document_search = partial(from_thread.run, document_tools.search_documents)
            chat_bot = FoundryChatBot(
                endpoint=settings.endpoint,
                model_name=settings.model_name,
                agent_name=settings.agent_name,
                document_search=document_search,
            )
            stack.callback(chat_bot.close)
            app.state.chat_bot = chat_bot
        else:
            credential = stack.enter_context(DefaultAzureCredential())
            model = ChatOpenAI(
                model=settings.model_name,
                base_url=f"{settings.endpoint.rstrip('/')}/openai/v1",
                api_key=get_bearer_token_provider(
                    credential, "https://ai.azure.com/.default"
                ),
            )
            stack.callback(model.root_client.close)
            await stack.enter_async_context(model.root_async_client)
            app.state.chat_bot = LangChainChatBot(model=model)
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    if body.history is None:
        answer = request.app.state.chat_bot.chat(body.message)
    else:
        history = [
            HistoryMessage(role=item.role, content=item.content) for item in body.history
        ]
        answer = request.app.state.chat_bot.chat_with_history(body.message, history)
    return ChatResponse(answer=answer)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
