"""Chatbot-side client contract for the team's future MCP service."""

import json

from mcp import ClientSession
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class DocumentPassage(BaseModel):
    model_config = ConfigDict(strict=True)

    chunk_id: str = Field(min_length=1)
    document: str = Field(min_length=1)
    pages: list[int]
    text: str = Field(min_length=1)


PASSAGES = TypeAdapter(list[DocumentPassage])


class McpDocumentSearch:
    """Expect search_documents(question) to return a JSON array of passages."""

    def __init__(self, session: ClientSession, tool_name: str = "search_documents"):
        self._session = session
        self._tool_name = tool_name

    async def search_documents(self, question: str) -> list[DocumentPassage]:
        if not question.strip():
            raise ValueError("Search question cannot be blank.")
        result = await self._session.call_tool(
            self._tool_name, arguments={"question": question},
        )
        if result.isError:
            raise RuntimeError("The MCP document-search tool reported an error.")
        # The proposed contract uses one JSON text block, or structured {passages: [...]}.
        if result.structuredContent is not None:
            if "passages" not in result.structuredContent:
                raise ValueError("MCP search structured output must contain 'passages'.")
            payload = result.structuredContent["passages"]
        else:
            if len(result.content) != 1 or result.content[0].type != "text":
                raise ValueError("MCP search must return one JSON text block.")
            payload = json.loads(result.content[0].text)
        return PASSAGES.validate_python(payload)
