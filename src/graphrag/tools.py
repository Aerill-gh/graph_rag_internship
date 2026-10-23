from typing import TypedDict

from neo4j import ManagedTransaction

from graphrag.models import generate_embedding
from graphrag.neo4j_store import create_neo4j_driver


VECTOR_INDEX_NAME = "chunk_embedding_index"
TOP_K = 3


class SearchResult(TypedDict):
    chunk_id: str
    document: str
    pages: list[int]
    text: str
    score: float


def _search_chunks(
    tx: ManagedTransaction,
    embedding: list[float],
) -> list[SearchResult]:
    records = tx.run(
        """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        RETURN
            node.chunk_id AS chunk_id,
            node.source_document AS document,
            node.pages AS pages,
            node.text AS text,
            score
        ORDER BY score DESC
        """,
        index_name=VECTOR_INDEX_NAME,
        top_k=TOP_K,
        embedding=embedding,
    )

    return [
        SearchResult(
            chunk_id=record["chunk_id"],
            document=record["document"],
            pages=record["pages"],
            text=record["text"],
            score=record["score"],
        )
        for record in records
    ]


def search_documents(question: str) -> list[SearchResult]:
    """Find Zava document passages relevant to a question.

    Return matching text, document names, page numbers, chunk IDs,
    and similarity scores so answers can cite their sources.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    embedding = generate_embedding(question)

    with create_neo4j_driver() as driver:
        with driver.session(database="neo4j") as session:
            return session.execute_read(_search_chunks, embedding)
