
import json

from neo4j import GraphDatabase

from graphrag.config import settings
from graphrag.ontology import ALLOWED_RELATIONSHIPS, NODE_TYPES


EMBEDDING_DIMENSIONS = 1536
VECTOR_INDEX_NAME = "chunk_embedding_index"
ENTITY_LABELS = NODE_TYPES - {"Document", "Chunk"}


def create_neo4j_schema(session, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
    """Create source constraints and the cosine index used by retrieval."""
    if type(dimensions) is not int or dimensions <= 0:
        raise ValueError("Embedding dimensions must be a positive integer.")

    session.run(
        "CREATE CONSTRAINT document_name IF NOT EXISTS "
        "FOR (document:Document) REQUIRE document.name IS UNIQUE"
    ).consume()
    session.run(
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
        "FOR (chunk:Chunk) REQUIRE chunk.chunk_id IS UNIQUE"
    ).consume()
    for label in sorted(ENTITY_LABELS):
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
            f"FOR (entity:{label}) REQUIRE entity.name IS UNIQUE"
        ).consume()
    session.run(
        "CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS "
        "FOR (chunk:Chunk) ON (chunk.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: "
        f"{dimensions}, `vector.similarity_function`: 'cosine'}}}}"
    ).consume()
    session.run(
        "CALL db.awaitIndex($index_name, $timeout_seconds)",
        index_name=VECTOR_INDEX_NAME,
        timeout_seconds=60,
    ).consume()


def store_chunk(session, chunk: dict, embedding: list[float]) -> None:
    """Store a document chunk and its embedding in Neo4j."""
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Embedding must have {EMBEDDING_DIMENSIONS} dimensions.")
    for field in ("chunk_id", "source_document", "text"):
        if not isinstance(chunk.get(field), str) or not chunk[field].strip():
            raise ValueError(f"Chunk requires a nonempty {field}.")

    session.run(
        """
        MERGE (d:Document {name: $source_document})
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.text = $text,
            c.source_document = $source_document,
            c.pages = $pages,
            c.token_count = $token_count,
            c.figures_json = $figures_json,
            c.embedding = $embedding,
            c.embedding_model = $embedding_model
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        chunk_id=chunk["chunk_id"],
        text=chunk["text"],
        source_document=chunk["source_document"],
        pages=chunk["pages"],
        token_count=chunk["token_count"],
        figures_json=json.dumps(chunk.get("figures", []), allow_nan=False),
        embedding=embedding,
        embedding_model=settings.azure_openai_embedding_deployment,
    ).consume()


def store_extraction(session, chunk_id: str, extraction: dict) -> None:
    """Atomically store extracted entities, MENTIONS links, and source-backed facts."""
    from graphrag.graph_extraction import validate_extraction

    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("A nonempty source chunk ID is required.")
    validate_extraction(extraction)
    entities = extraction["entities"]
    relationships = extraction["relationships"]

    def persist(transaction):
        if transaction.run(
            "MATCH (chunk:Chunk {chunk_id: $chunk_id}) RETURN chunk.chunk_id AS chunk_id",
            chunk_id=chunk_id,
        ).single() is None:
            raise ValueError(f"Source chunk {chunk_id} must be stored before its extraction.")

        for entity in entities:
            label = entity["label"]
            if label not in ENTITY_LABELS:
                raise ValueError(f"Unsupported entity label: {label}")
            transaction.run(
                f"MATCH (chunk:Chunk {{chunk_id: $chunk_id}}) "
                f"MERGE (entity:{label} {{name: $name}}) "
                "MERGE (chunk)-[:MENTIONS]->(entity)",
                chunk_id=chunk_id, name=entity["name"],
            ).consume()

        for relationship in relationships:
            source_label = relationship["source_label"]
            target_label = relationship["target_label"]
            relation_type = relationship["type"]
            if (source_label, relation_type, target_label) not in ALLOWED_RELATIONSHIPS:
                raise ValueError("Unsupported relationship pattern.")
            transaction.run(
                f"MATCH (source:{source_label} {{name: $source_name}}), "
                f"(target:{target_label} {{name: $target_name}}) "
                f"MERGE (source)-[fact:{relation_type}]->(target) "
                "SET fact.source_chunk_ids = CASE "
                "WHEN $chunk_id IN coalesce(fact.source_chunk_ids, []) "
                "THEN coalesce(fact.source_chunk_ids, []) "
                "ELSE coalesce(fact.source_chunk_ids, []) + [$chunk_id] END",
                source_name=relationship["source_name"],
                target_name=relationship["target_name"],
                chunk_id=chunk_id,
            ).consume()

    session.execute_write(persist)


def create_neo4j_driver():
    """Create a Neo4j driver using project settings."""
    neo4j_settings = settings.neo4j
    return GraphDatabase.driver(
        neo4j_settings.uri,
        auth=(neo4j_settings.username, neo4j_settings.password),
    )
