
import argparse
from functools import partial

from azure.identity import DefaultAzureCredential

from database_mcp.client import DatabaseMcpClient
from graphrag.components import (
    AzureBlobDocumentLoader,
    ZavaDocumentParser,
    ZavaDocumentSplitter,
    ZavaChunkEmbedder,
    McpChunkLoader,
)
from graphrag.pipeline import CompositeDocumentProcessingPipeline

from graphrag.documents import (
    parse_pdf_document,
    create_chunks,
    create_markdown_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the modular Zava document ingestion pipeline."
    )
    parser.add_argument(
        "--blob",
        required=True,
        help="Name of the document in Azure Blob Storage.",
    )
    args = parser.parse_args()

    credential = DefaultAzureCredential()

    try:
        with DatabaseMcpClient() as client:
            client.health_check()
            client.ensure_schema()
            pipeline = CompositeDocumentProcessingPipeline(
                document_loader=AzureBlobDocumentLoader(credential),
                document_parser=ZavaDocumentParser(
                    pdf_parser=partial(
                        parse_pdf_document,
                        credential=credential,
                    )
                ),
                document_splitter=ZavaDocumentSplitter(
                    pdf_splitter=create_chunks,
                    markdown_splitter=create_markdown_chunks,
                ),
                chunk_embedder=ZavaChunkEmbedder(),
                chunk_loader=McpChunkLoader(client),
            )

            pipeline.load_document(args.blob)
            print(f"Successfully processed: {args.blob}")

    finally:
        credential.close()


if __name__ == "__main__":
    main()
