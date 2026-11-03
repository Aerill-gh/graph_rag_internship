"""Local ingestion-only MCP endpoint; does not register chatbot tools."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from database_mcp.contracts import (
    ChunkReadResponse,
    ChunkWrite,
    ChunkWriteResponse,
    ExtractionWrite,
    ExtractionWriteResponse,
    HealthResponse,
    SchemaResponse,
)


class DatabaseBackend(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def health_check(self) -> HealthResponse: ...
    def ensure_schema(self) -> SchemaResponse: ...
    def upsert_chunk(self, request: ChunkWrite) -> ChunkWriteResponse: ...
    def upsert_extraction(self, request: ExtractionWrite) -> ExtractionWriteResponse: ...
    def get_chunk(self, chunk_id: str) -> ChunkReadResponse: ...


def create_mcp_server(backend: DatabaseBackend | None = None) -> FastMCP:
    active_backend = backend

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[None]:
        nonlocal active_backend
        if active_backend is None:
            from database_mcp.config import load_database_settings
            from database_mcp.neo4j_backend import Neo4jBackend

            active_backend = Neo4jBackend(load_database_settings())
        try:
            await to_thread.run_sync(active_backend.open)
            yield None
        finally:
            await to_thread.run_sync(active_backend.close)

    server = FastMCP(
        "Zava database ingestion",
        host="127.0.0.1",
        port=8002,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        lifespan=lifespan,
    )

    def database() -> DatabaseBackend:
        if active_backend is None:
            raise RuntimeError("Database MCP backend is unavailable.")
        return active_backend

    read_annotations = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False,
        idempotentHint=True, openWorldHint=False,
    )
    write_annotations = ToolAnnotations(
        readOnlyHint=False, destructiveHint=True,
        idempotentHint=True, openWorldHint=False,
    )

    @server.tool(annotations=read_annotations)
    async def health_check() -> HealthResponse:
        """Verify the MCP server can connect to its database."""
        return await to_thread.run_sync(database().health_check)

    @server.tool(annotations=write_annotations)
    async def ensure_schema() -> SchemaResponse:
        """Ensure source constraints and the existing 1536-dimensional vector index."""
        return await to_thread.run_sync(database().ensure_schema)

    @server.tool(annotations=write_annotations)
    async def upsert_chunk(request: ChunkWrite) -> ChunkWriteResponse:
        """Store a validated source chunk; preserve its vector if no embedding is supplied."""
        validated = ChunkWrite.model_validate(request.model_dump())
        return await to_thread.run_sync(database().upsert_chunk, validated)

    @server.tool(annotations=write_annotations)
    async def upsert_extraction(request: ExtractionWrite) -> ExtractionWriteResponse:
        """Atomically store allowed graph facts linked to an existing source chunk."""
        validated = ExtractionWrite.model_validate(request.model_dump())
        return await to_thread.run_sync(database().upsert_extraction, validated)

    @server.tool(annotations=read_annotations)
    async def get_chunk(chunk_id: str) -> ChunkReadResponse:
        """Read one source chunk, its vector, and up to 50 source-backed entities and facts."""
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Source chunk ID cannot be blank.")
        return await to_thread.run_sync(database().get_chunk, chunk_id)

    return server


if __name__ == "__main__":
    create_mcp_server().run(transport="streamable-http")
