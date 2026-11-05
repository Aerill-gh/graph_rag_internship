
import argparse
from functools import partial
from pathlib import PurePosixPath

from azure.identity import DefaultAzureCredential

from graphrag.database_mcp.client import DatabaseMcpClient
from graphrag.components import (
    LocalDocumentLoader,
    ZavaDocumentParser,
    ZavaDocumentSplitter,
    ZavaChunkEmbedder,
    McpChunkLoader,
)
from graphrag.pipeline import CompositeDocumentProcessingPipeline, DocumentSource

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
        "--document",
        required=True,
        help="Path relative to local document_storage.",
    )
    args = parser.parse_args()

    credential = DefaultAzureCredential()

    try:
        with DatabaseMcpClient() as client:
            client.health_check()
            client.ensure_schema()
            pipeline = CompositeDocumentProcessingPipeline(
                document_loader=LocalDocumentLoader(),
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

            source_document = PurePosixPath(args.document.replace("\\", "/")).as_posix()
            pipeline.load_document(DocumentSource(
                name=source_document, location=args.document,
            ))
            print(f"Successfully processed: {source_document}")

    finally:
        credential.close()


if __name__ == "__main__":
    main()
