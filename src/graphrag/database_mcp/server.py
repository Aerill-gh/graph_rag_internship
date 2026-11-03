"""Local ingestion-only MCP endpoint; does not register chatbot tools."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal, Protocol

from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .contracts import (
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


class DatabaseBackend(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def health_check(self) -> HealthResponse: ...
    def ensure_schema(self) -> SchemaResponse: ...
    def upsert_chunk(self, request: ChunkWrite) -> ChunkWriteResponse: ...
    def upsert_extraction(self, request: ExtractionWrite) -> ExtractionWriteResponse: ...
    def get_chunk(self, chunk_id: str) -> ChunkReadResponse: ...
    def search_documents(
        self, request: SearchDocumentsRequest,
    ) -> SearchDocumentsResponse: ...
    def find_entities(self, request: FindEntitiesRequest) -> FindEntitiesResponse: ...
    def explore_entity(self, request: ExploreEntityRequest) -> ExploreEntityResponse: ...
    def query_graph(self, request: QueryGraphRequest) -> QueryGraphResponse: ...
    def corpus_summary(self) -> CorpusSummaryResponse: ...


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

    @server.tool(annotations=read_annotations)
    async def search_documents(
        embedding: list[float], top_k: int = 5,
    ) -> SearchDocumentsResponse:
        """Search indexed document chunks using a 1536-dimensional query embedding."""
        request = SearchDocumentsRequest(embedding=embedding, top_k=top_k)
        return await to_thread.run_sync(database().search_documents, request)

    @server.tool(annotations=read_annotations)
    async def find_entities(name: str, limit: int = 10) -> FindEntitiesResponse:
        """Find allowed ontology entities by a case-insensitive name substring."""
        request = FindEntitiesRequest(name=name, limit=limit)
        return await to_thread.run_sync(database().find_entities, request)

    @server.tool(annotations=read_annotations)
    async def explore_entity(
        name: str, label: str, limit: int = 20,
    ) -> ExploreEntityResponse:
        """Read bounded, source-backed facts attached to one named graph entity."""
        request = ExploreEntityRequest(name=name, label=label, limit=limit)
        return await to_thread.run_sync(database().explore_entity, request)

    @server.tool(annotations=read_annotations)
    async def query_graph(
        pattern: str,
        source_names: list[str],
        target_name: str = "",
        match_mode: Literal["any", "all", "difference"] = "any",
        limit: int = 20,
    ) -> QueryGraphResponse:
        """Query bounded graph facts using an allowlisted ontology pattern, never raw Cypher."""
        request = QueryGraphRequest(
            pattern=pattern, source_names=source_names, target_name=target_name,
            match_mode=match_mode, limit=limit,
        )
        return await to_thread.run_sync(database().query_graph, request)

    @server.tool(annotations=read_annotations)
    async def corpus_summary() -> CorpusSummaryResponse:
        """Return bounded corpus counts and a digest without exposing full graph contents."""
        return await to_thread.run_sync(database().corpus_summary)

    return server


if __name__ == "__main__":
    create_mcp_server().run(transport="streamable-http")
