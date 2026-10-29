
from neo4j import GraphDatabase

from graphrag.config import settings
from graphrag.models import generate_embedding


def main():
    question = "What are the employee benefits?"
    question_embedding = generate_embedding(question)

    neo4j_settings = settings.neo4j
    with GraphDatabase.driver(
        neo4j_settings.uri,
        auth=(neo4j_settings.username, neo4j_settings.password),
    ) as driver:
        with driver.session(database=neo4j_settings.database) as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes(
                    'chunk_embedding_index',
                    3,
                    $embedding
                )
                YIELD node, score
                RETURN
                    node.chunk_id AS chunk_id,
                    node.source_document AS document,
                    score
                ORDER BY score DESC
                """,
                embedding=question_embedding,
            )

            for record in result:
                print(
                    f"Chunk: {record['chunk_id']} | "
                    f"Document: {record['document']} | "
                    f"Score: {record['score']:.4f}"
                )


if __name__ == "__main__":
    main()
