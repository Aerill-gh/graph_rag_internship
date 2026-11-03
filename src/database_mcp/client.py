"""Synchronous ingestion adapter over one persistent async MCP session."""

import argparse
import json
from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from functools import partial
from types import TracebackType
from typing import Any, TypeVar

from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel

from database_mcp.config import database_mcp_url
from database_mcp.contracts import (
    ChunkReadResponse,
    ChunkWrite,
    ChunkWriteResponse,
    ExtractionWrite,
    ExtractionWriteResponse,
    HealthResponse,
    SchemaResponse,
)


Response = TypeVar("Response", bound=BaseModel)


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
            raise RuntimeError(
                f"Database MCP tool {name} failed. Check the database MCP server."
            )
        if result.structuredContent is not None:
            payload = result.structuredContent
        else:
            if len(result.content) != 1 or result.content[0].type != "text":
                raise ValueError(f"Database MCP tool {name} returned invalid content.")
            payload = json.loads(result.content[0].text)
        return response_type.model_validate(payload)

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
