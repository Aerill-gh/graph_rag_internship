"""Synchronous ingestion adapter over one persistent async MCP session."""

import argparse
import json
from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from functools import partial
from types import TracebackType
from typing import Any, Literal, TypeVar

from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel

from database_mcp.config import database_mcp_url
from database_mcp.contracts import (
    ChunkReadResponse,
    ChunkWrite,
    ChunkWriteResponse,
    CorpusSummaryResponse,
    ExploreEntityRequest,
    ExploreEntityResponse,
    ExtractionWrite,
    ExtractionWriteResponse,
    FindEntitiesRequest,
    FindEntitiesResponse,
    HealthResponse,
    QueryGraphRequest,
    QueryGraphResponse,
    SchemaResponse,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
)


Response = TypeVar("Response", bound=BaseModel)


class DatabaseMcpError(RuntimeError):
    """The database MCP service failed or returned an invalid tool response."""


class DatabaseMcpClient:
    def __init__(self, url: str | None = None) -> None:
        self.url = url if url is not None else database_mcp_url()
        self._stack: ExitStack | None = None
        self._portal: BlockingPortal | None = None
        self._session: ClientSession | None = None

    @asynccontextmanager
    async def _open_session(self) -> AsyncIterator[ClientSession]:
        async with streamable_http_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    def __enter__(self) -> "DatabaseMcpClient":
        if self._stack is not None:
            raise RuntimeError("Database MCP client is already open.")
        with ExitStack() as stack:
            portal = stack.enter_context(start_blocking_portal())
            # MCP transport task groups must enter and exit in the same async task.
            session = stack.enter_context(
                portal.wrap_async_context_manager(self._open_session())
            )
            self._portal = portal
            self._session = session
            self._stack = stack.pop_all()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._stack is not None:
                self._stack.__exit__(exc_type, exc_value, traceback)
        finally:
            self._stack = None
            self._portal = None
            self._session = None

    def _call(
        self, name: str, arguments: dict[str, Any], response_type: type[Response],
    ) -> Response:
        if self._portal is None or self._session is None:
            raise RuntimeError("Use DatabaseMcpClient inside a with block.")
        result = self._portal.call(
            partial(self._session.call_tool, name, arguments=arguments)
        )
        if result.isError:
            raise DatabaseMcpError(
                f"Database MCP tool {name} failed. Check the database MCP server."
            )
        try:
            if result.structuredContent is not None:
                payload = result.structuredContent
            else:
                if len(result.content) != 1 or result.content[0].type != "text":
                    raise ValueError("MCP response must contain one JSON text block.")
                payload = json.loads(result.content[0].text)
            return response_type.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise DatabaseMcpError(
                f"Database MCP tool {name} returned an invalid response."
            ) from exc

    def health_check(self) -> HealthResponse:
        return self._call("health_check", {}, HealthResponse)

    def ensure_schema(self) -> SchemaResponse:
        return self._call("ensure_schema", {}, SchemaResponse)

    def store_chunk(
        self, chunk: dict, embedding: list[float] | None = None,
        *, embedding_model: str | None = None,
    ) -> None:
        request = ChunkWrite.model_validate({
            "chunk": chunk, "embedding": embedding, "embedding_model": embedding_model,
        })
        response = self._call(
            "upsert_chunk", {"request": request.model_dump()}, ChunkWriteResponse,
        )
        if response.chunk_id != request.chunk.chunk_id:
            raise ValueError("Database MCP acknowledged a different source chunk.")

    def store_extraction(self, chunk_id: str, extraction: dict) -> None:
        request = ExtractionWrite.model_validate({
            "chunk_id": chunk_id, "extraction": extraction,
        })
        response = self._call(
            "upsert_extraction", {"request": request.model_dump()}, ExtractionWriteResponse,
        )
        if (
            response.chunk_id != chunk_id
            or response.entity_mentions != len(request.extraction.entities)
            or response.relationship_mentions != len(request.extraction.relationships)
        ):
            raise ValueError("Database MCP extraction acknowledgement does not match the request.")

    def get_chunk(self, chunk_id: str) -> ChunkReadResponse:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Source chunk ID cannot be blank.")
        response = self._call("get_chunk", {"chunk_id": chunk_id}, ChunkReadResponse)
        if response.chunk.chunk_id != chunk_id:
            raise ValueError("Database MCP returned a different source chunk.")
        return response

    def search_documents(
        self, embedding: list[float], top_k: int = 5,
    ) -> SearchDocumentsResponse:
        request = SearchDocumentsRequest(embedding=embedding, top_k=top_k)
        return self._call("search_documents", request.model_dump(), SearchDocumentsResponse)

    def find_entities(self, name: str, limit: int = 10) -> FindEntitiesResponse:
        request = FindEntitiesRequest(name=name, limit=limit)
        return self._call("find_entities", request.model_dump(), FindEntitiesResponse)

    def explore_entity(
        self, name: str, label: str, limit: int = 20,
    ) -> ExploreEntityResponse:
        request = ExploreEntityRequest(name=name, label=label, limit=limit)
        return self._call("explore_entity", request.model_dump(), ExploreEntityResponse)

    def query_graph(
        self,
        pattern: str,
        source_names: list[str],
        target_name: str = "",
        match_mode: Literal["any", "all", "difference"] = "any",
        limit: int = 20,
    ) -> QueryGraphResponse:
        request = QueryGraphRequest(
            pattern=pattern, source_names=source_names, target_name=target_name,
            match_mode=match_mode, limit=limit,
        )
        return self._call("query_graph", request.model_dump(), QueryGraphResponse)

    def corpus_summary(self) -> CorpusSummaryResponse:
        return self._call("corpus_summary", {}, CorpusSummaryResponse)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the ingestion MCP server or read one chunk.")
    parser.add_argument("--chunk-id", help="Read an existing chunk and its source-backed facts.")
    args = parser.parse_args()
    with DatabaseMcpClient() as client:
        client.health_check()
        print("Database MCP connection verified.")
        if args.chunk_id:
            print(client.get_chunk(args.chunk_id).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
