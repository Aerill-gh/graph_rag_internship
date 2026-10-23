import json

from graphrag.config import settings
from graphrag.embeddings import generate_embedding
from graphrag.neo4j_store import VECTOR_INDEX_NAME, create_neo4j_driver


SEARCH_DOCUMENTS_DESCRIPTION = (
    "Search the indexed Zava document chunks for evidence relevant to a question. "
    "Returns ranked source passages with chunk ID, document, pages, and similarity. "
    "Use the returned passages as evidence and cite their source. Results may not be exhaustive."
)


SEARCH_CHUNKS_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node AS chunk, score
RETURN chunk.chunk_id AS chunk_id,
       chunk.source_document AS source_document,
       chunk.pages AS pages,
       chunk.text AS text,
       chunk.figures_json AS figures_json,
       score
ORDER BY score DESC
"""


def search_documents(question: str, top_k: int = 5) -> list[dict]:
    """Embed a question and return source-linked chunks from Neo4j."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string.")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100.")

    embedding = generate_embedding(question)
    if not embedding:
        raise ValueError("Question embedding must not be empty.")

    with create_neo4j_driver() as driver:
        with driver.session(database=settings.neo4j.database) as session:
            rows = session.run(
                SEARCH_CHUNKS_QUERY,
                index_name=VECTOR_INDEX_NAME,
                top_k=top_k,
                embedding=embedding,
            ).data()

    passages = []
    for row in rows:
        passage = dict(row)
        passage["figures"] = json.loads(passage.pop("figures_json", None) or "[]")
        passages.append(passage)
    return passages