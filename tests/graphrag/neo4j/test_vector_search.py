
from database_mcp.client import DatabaseMcpClient
from graphrag.models import generate_embedding


def main():
    question = "What are the employee benefits?"
    question_embedding = generate_embedding(question)

    with DatabaseMcpClient() as client:
        result = client.search_documents(question_embedding, top_k=3)
    for passage in result.passages:
        print(
            f"Chunk: {passage.chunk_id} | "
            f"Document: {passage.source_document} | "
            f"Score: {passage.score:.4f}"
        )


if __name__ == "__main__":
    main()
