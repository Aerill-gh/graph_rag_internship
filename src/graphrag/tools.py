from typing import TypedDict

from database_mcp.client import DatabaseMcpClient
from graphrag.models import generate_embedding


TOP_K = 3


class SearchResult(TypedDict):
    chunk_id: str
    document: str
    pages: list[int]
    text: str
    score: float


def search_documents(question: str) -> list[SearchResult]:
    """Find Zava document passages relevant to a question.

    Return matching text, document names, page numbers, chunk IDs,
    and similarity scores so answers can cite their sources.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    embedding = generate_embedding(question)
    if not embedding:
        raise ValueError("Question embedding must not be empty.")

    with DatabaseMcpClient() as client:
        response = client.search_documents(embedding, TOP_K)
    return [
        SearchResult(
            chunk_id=passage.chunk_id,
            document=passage.source_document,
            pages=passage.pages,
            text=passage.text,
            score=passage.score,
        )
        for passage in response.passages
    ]
