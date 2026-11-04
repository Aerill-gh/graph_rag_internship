"""Neo4j implementation used only by the ingestion MCP server."""

import hashlib
import json

from neo4j import Driver, GraphDatabase, ManagedTransaction

from .config import DatabaseSettings
from .contracts import (
    ChunkPayload,
    ChunkReadResponse,
    ChunkWrite,
    ChunkWriteResponse,
    CorpusSummaryResponse,
    DOMAIN_PATTERNS,
    ENTITY_LABELS,
    EntityCandidate,
    ExploreEntityRequest,
    ExploreEntityResponse,
    Entity,
    ExtractionWrite,
    ExtractionWriteResponse,
    FindEntitiesRequest,
    FindEntitiesResponse,
    GraphFact,
    HealthResponse,
    Relationship,
    QueryGraphRequest,
    QueryGraphResponse,
    SchemaResponse,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
    SearchPassage,
)
from graphrag.Neo4j.neo4j_store import create_neo4j_schema, store_chunk, store_extraction


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

    def search_documents(self, request: SearchDocumentsRequest) -> SearchDocumentsResponse:
        query = (
            "CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) "
            "YIELD node AS chunk, score "
            "RETURN chunk.chunk_id AS chunk_id, "
            "chunk.source_document AS source_document, chunk.pages AS pages, "
            "chunk.text AS text, chunk.figures_json AS figures_json, score "
            "ORDER BY score DESC"
        )

        def read(transaction: ManagedTransaction) -> list[dict]:
            return transaction.run(
                query,
                index_name="chunk_embedding_index",
                top_k=request.top_k,
                embedding=request.embedding,
            ).data()

        with self._driver().session(database=self.configuration.database) as session:
            rows = session.execute_read(read)
        passages = [
            SearchPassage(
                chunk_id=row["chunk_id"],
                source_document=row["source_document"],
                pages=row["pages"],
                text=row["text"],
                figures=json.loads(row.get("figures_json") or "[]"),
                score=row["score"],
            )
            for row in rows
        ]
        return SearchDocumentsResponse(passages=passages)

    def find_entities(self, request: FindEntitiesRequest) -> FindEntitiesResponse:
        query = (
            "MATCH (entity) WHERE any(label IN labels(entity) WHERE label IN $labels) "
            "AND toLower(entity.name) CONTAINS toLower($name) "
            "RETURN labels(entity) AS labels, entity.name AS name "
            "ORDER BY name LIMIT $limit"
        )

        def read(transaction: ManagedTransaction) -> list[dict]:
            return transaction.run(
                query, labels=sorted(ENTITY_LABELS), name=request.name.strip(),
                limit=request.limit,
            ).data()

        with self._driver().session(database=self.configuration.database) as session:
            rows = session.execute_read(read)
        return FindEntitiesResponse(candidates=[EntityCandidate.model_validate(row) for row in rows])

    def explore_entity(self, request: ExploreEntityRequest) -> ExploreEntityResponse:
        query = (
            "MATCH (source)-[fact]->(target) "
            "WHERE (source.name = $name AND $label IN labels(source)) "
            "OR (target.name = $name AND $label IN labels(target)) "
            "WITH source, fact, target WHERE size(coalesce(fact.source_chunk_ids, [])) > 0 "
            "UNWIND fact.source_chunk_ids AS chunk_id "
            "MATCH (chunk:Chunk {chunk_id: chunk_id}) "
            "RETURN labels(source) AS source_labels, source.name AS source, "
            "type(fact) AS relationship, labels(target) AS target_labels, target.name AS target, "
            "chunk.chunk_id AS chunk_id, chunk.source_document AS document, "
            "chunk.pages AS pages, chunk.text AS text "
            "ORDER BY source, relationship, target, chunk_id LIMIT $limit"
        )

        def read(transaction: ManagedTransaction) -> list[dict]:
            return transaction.run(
                query, name=request.name.strip(), label=request.label, limit=request.limit,
            ).data()

        with self._driver().session(database=self.configuration.database) as session:
            rows = session.execute_read(read)
        return ExploreEntityResponse(
            facts=[GraphFact.model_validate(row) for row in rows],
            truncated=len(rows) == request.limit,
        )

    def query_graph(self, request: QueryGraphRequest) -> QueryGraphResponse:
        pattern = next(
            pattern for pattern in DOMAIN_PATTERNS
            if "|".join(pattern) == request.pattern
        )
        source_label, relationship, target_label = pattern
        conditions = [
            "($target_name = '' OR toLower(target.name) CONTAINS toLower($target_name))",
            "size(coalesce(fact.source_chunk_ids, [])) > 0",
        ]
        if request.match_mode == "difference":
            conditions.extend([
                "toLower(source.name) CONTAINS toLower($source_names[0])",
                f"NOT EXISTS {{ MATCH (other:{source_label})-[:{relationship}]->(target) "
                "WHERE toLower(other.name) CONTAINS toLower($source_names[1]) }",
            ])
        else:
            conditions.append(
                "any(name IN $source_names WHERE toLower(source.name) CONTAINS toLower(name))"
            )
            if request.match_mode == "all":
                conditions.append(
                    "all(name IN $source_names WHERE EXISTS { "
                    f"MATCH (other:{source_label})-[other_fact:{relationship}]->(target) "
                    "WHERE toLower(other.name) CONTAINS toLower(name) "
                    "AND size(coalesce(other_fact.source_chunk_ids, [])) > 0 })"
                )
        cypher = (
            f"MATCH (source:{source_label})-[fact:{relationship}]->(target:{target_label}) "
            "WHERE " + " AND ".join(conditions) + " "
            "UNWIND fact.source_chunk_ids AS chunk_id "
            "MATCH (chunk:Chunk {chunk_id: chunk_id}) "
            "RETURN labels(source) AS source_labels, source.name AS source, "
            "type(fact) AS relationship, labels(target) AS target_labels, target.name AS target, "
            "chunk.chunk_id AS chunk_id, chunk.source_document AS document, "
            "chunk.pages AS pages, chunk.text AS text "
            "ORDER BY source, target, chunk_id LIMIT $limit"
        )
        parameters = {
            "source_names": [name.strip() for name in request.source_names],
            "target_name": request.target_name.strip(),
            "limit": request.limit + 1,
        }

        def read(transaction: ManagedTransaction) -> list[dict]:
            return transaction.run(cypher, **parameters).data()

        with self._driver().session(database=self.configuration.database) as session:
            rows = session.execute_read(read)
        return QueryGraphResponse(
            facts=[GraphFact.model_validate(row) for row in rows[:request.limit]],
            cypher=cypher,
            parameters=parameters,
            result_limit=request.limit,
            truncated=len(rows) > request.limit,
            scope="Extracted graph facts only; absence is not evidence of exclusion.",
        )

    def corpus_summary(self) -> CorpusSummaryResponse:
        record_limit = 10_001

        def read(transaction: ManagedTransaction) -> tuple[list[dict], list[dict]]:
            nodes = transaction.run(
                "MATCH (node) RETURN labels(node) AS labels, properties(node) AS properties "
                "LIMIT $limit",
                limit=record_limit,
            ).data()
            relationships = transaction.run(
                "MATCH (source)-[fact]->(target) "
                "RETURN labels(source) AS source_labels, "
                "coalesce(source.chunk_id, source.name) AS source, type(fact) AS type, "
                "labels(target) AS target_labels, "
                "coalesce(target.chunk_id, target.name) AS target, properties(fact) AS properties "
                "LIMIT $limit",
                limit=record_limit,
            ).data()
            return nodes, relationships

        with self._driver().session(database=self.configuration.database) as session:
            nodes, relationships = session.execute_read(read)
        if len(nodes) == record_limit or len(relationships) == record_limit:
            raise ValueError("Corpus summary exceeds the 10000-record limit.")
        chunks = sum("Chunk" in node["labels"] for node in nodes)
        canonical = {
            "nodes": sorted(json.dumps(node, sort_keys=True) for node in nodes),
            "relationships": sorted(
                json.dumps(fact, sort_keys=True) for fact in relationships
            ),
        }
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        return CorpusSummaryResponse(
            database=self.configuration.database,
            chunks=chunks,
            nodes=len(nodes),
            relationships=len(relationships),
            sha256=digest,
        )

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
