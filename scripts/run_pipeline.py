
import argparse
from functools import partial

from azure.identity import DefaultAzureCredential

from graphrag.components import (
    AzureBlobDocumentLoader,
    ZavaDocumentParser,
    ZavaDocumentSplitter,
    ZavaChunkEmbedder,
    Neo4jChunkLoader,
)
from graphrag.neo4j_store import create_neo4j_driver
from graphrag.pipeline import CompositeDocumentProcessingPipeline

from scripts.chunk_pdf import (
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
        with create_neo4j_driver() as driver:
            driver.verify_connectivity()

            with driver.session(database="neo4j") as session:
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
                    chunk_loader=Neo4jChunkLoader(session),
                )

                pipeline.load_document(args.blob)
                print(f"Successfully processed: {args.blob}")

    finally:
        credential.close()


if __name__ == "__main__":
    main()
