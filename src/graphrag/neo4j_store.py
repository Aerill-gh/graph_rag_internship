
import json

from neo4j import GraphDatabase

from graphrag.config import settings


EMBEDDING_DIMENSIONS = 1536
VECTOR_INDEX_NAME = "chunk_embedding_index"


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


def create_neo4j_driver():
    """Create a Neo4j driver using project settings."""
    neo4j_settings = settings.neo4j
    return GraphDatabase.driver(
        neo4j_settings.uri,
        auth=(neo4j_settings.username, neo4j_settings.password),
    )
