"""Expose the team's retrieval function through a local MCP endpoint."""

from collections.abc import Callable

from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from Chatbot.document_tools import DocumentPassage


class SearchResponse(BaseModel):
    passages: list[DocumentPassage]


def retrieve_documents(question: str) -> list[DocumentPassage]:
    # Load retrieval only when called, so MCP discovery needs no Azure credentials.
    from graphrag.tools import search_documents

    return [DocumentPassage.model_validate(result) for result in search_documents(question)]


def create_mcp_server(
    document_search: Callable[[str], list[DocumentPassage]] = retrieve_documents,
) -> FastMCP:
    server = FastMCP(
        "Zava document search",
        host="127.0.0.1",
        port=8001,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
    )

    @server.tool(
        name="search_documents",
        description=(
            "Search Zava documents for passages about employee roles, policies, "
            "and benefits. Returns source text, document names, pages, and chunk IDs. "
            "An empty passages list means no matching evidence was found."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def search_documents(question: str) -> SearchResponse:
        if not question.strip():
            raise ValueError("Search question cannot be blank.")
        # Embedding and Neo4j calls are synchronous; keep them off the event loop.
        passages = await to_thread.run_sync(document_search, question)
        return SearchResponse(passages=passages)

    return server


if __name__ == "__main__":
    create_mcp_server().run(transport="streamable-http")
