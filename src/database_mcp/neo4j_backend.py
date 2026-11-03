"""Neo4j implementation used only by the ingestion MCP server."""

import json

from neo4j import Driver, GraphDatabase, ManagedTransaction

from database_mcp.config import DatabaseSettings
from database_mcp.contracts import (
    ChunkPayload,
    ChunkReadResponse,
    ChunkWrite,
    ChunkWriteResponse,
    Entity,
    ExtractionWrite,
    ExtractionWriteResponse,
    HealthResponse,
    Relationship,
    SchemaResponse,
)
from Neo4j.neo4j_store import create_neo4j_schema, store_chunk, store_extraction


class Neo4jBackend:
    def __init__(self, configuration: DatabaseSettings) -> None:
        self.configuration = configuration
        self.driver: Driver | None = None

    def open(self) -> None:
        if self.driver is not None:
            raise RuntimeError("Database backend is already open.")
        self.driver = GraphDatabase.driver(
            self.configuration.uri,
            auth=(self.configuration.username, self.configuration.password),
        )

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def _driver(self) -> Driver:
        if self.driver is None:
            raise RuntimeError("Database backend is not open.")
        return self.driver

    def health_check(self) -> HealthResponse:
        self._driver().verify_connectivity()
        return HealthResponse()

    def ensure_schema(self) -> SchemaResponse:
        with self._driver().session(database=self.configuration.database) as session:
            create_neo4j_schema(session)
        return SchemaResponse()

    def upsert_chunk(self, request: ChunkWrite) -> ChunkWriteResponse:
        with self._driver().session(database=self.configuration.database) as session:
            session.execute_write(
                lambda transaction: store_chunk(
                    transaction, request.chunk.model_dump(), request.embedding,
                    embedding_model=request.embedding_model,
                )
            )
        return ChunkWriteResponse(chunk_id=request.chunk.chunk_id)

    def upsert_extraction(self, request: ExtractionWrite) -> ExtractionWriteResponse:
        with self._driver().session(database=self.configuration.database) as session:
            store_extraction(session, request.chunk_id, request.extraction.model_dump())
        return ExtractionWriteResponse(
            chunk_id=request.chunk_id,
            entity_mentions=len(request.extraction.entities),
            relationship_mentions=len(request.extraction.relationships),
        )

    def get_chunk(self, chunk_id: str) -> ChunkReadResponse:
        def read(transaction: ManagedTransaction) -> ChunkReadResponse:
            record = transaction.run(
                "MATCH (chunk:Chunk {chunk_id: $chunk_id}) "
                "RETURN properties(chunk) AS chunk",
                chunk_id=chunk_id,
            ).single()
            if record is None:
                raise ValueError("Source chunk was not found.")
            chunk = record["chunk"]
            entities = transaction.run(
                "MATCH (:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(entity) "
                "UNWIND labels(entity) AS label "
                "RETURN label, entity.name AS name ORDER BY label, name LIMIT 51",
                chunk_id=chunk_id,
            ).data()
            relationships = transaction.run(
                "MATCH (source)-[fact]->(target) "
                "WHERE $chunk_id IN coalesce(fact.source_chunk_ids, []) "
                "UNWIND labels(source) AS source_label "
                "UNWIND labels(target) AS target_label "
                "RETURN source_label, source.name AS source_name, type(fact) AS type, "
                "target_label, target.name AS target_name "
                "ORDER BY source_label, source_name, type, target_name LIMIT 51",
                chunk_id=chunk_id,
            ).data()
            return ChunkReadResponse(
                chunk=ChunkPayload(
                    chunk_id=chunk["chunk_id"],
                    source_document=chunk["source_document"],
                    text=chunk["text"],
                    pages=chunk["pages"],
                    token_count=chunk["token_count"],
                    figures=json.loads(chunk.get("figures_json") or "[]"),
                ),
                embedding=chunk.get("embedding"),
                embedding_model=chunk.get("embedding_model"),
                entities=[Entity.model_validate(row) for row in entities[:50]],
                relationships=[
                    Relationship.model_validate(row) for row in relationships[:50]
                ],
                entities_truncated=len(entities) > 50,
                relationships_truncated=len(relationships) > 50,
            )

        with self._driver().session(database=self.configuration.database) as session:
            return session.execute_read(read)
