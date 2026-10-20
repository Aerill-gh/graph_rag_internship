
from neo4j import GraphDatabase

from graphrag.config import settings


def store_chunk(session, chunk: dict, embedding: list[float]) -> None:
    """Store a document chunk and its embedding in Neo4j."""
    if len(embedding) != 1536:
        raise ValueError("Embedding must have 1536 dimensions.")

    session.run(
        """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.text = $text,
            c.source_document = $source_document,
            c.pages = $pages,
            c.token_count = $token_count,
            c.embedding = $embedding,
            c.embedding_model = $embedding_model
        """,
        chunk_id=chunk["chunk_id"],
        text=chunk["text"],
        source_document=chunk["source_document"],
        pages=chunk["pages"],
        token_count=chunk["token_count"],
        embedding=embedding,
        embedding_model=settings.embedding_deployment,
    ).consume()


def create_neo4j_driver():
    """Create a Neo4j driver using project settings."""
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
