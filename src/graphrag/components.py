from collections.abc import Callable
from pathlib import PurePosixPath

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
from neo4j import Session

from graphrag.config import settings
from graphrag.embeddings import get_chunk_embedding
from graphrag.neo4j_store import store_chunk
from graphrag.pipeline import (
    ChunkEmbedder,
    ChunkLoader,
    DocumentLoader,
    DocumentParser,
    DocumentSplitter,
    ParsedDocument,
)


class AzureBlobDocumentLoader(DocumentLoader):
    """Load documents from Azure Blob Storage."""

    def __init__(self, credential: DefaultAzureCredential) -> None:
        self.credential = credential

    def load(self, document_path: str) -> bytes:
        with BlobClient(
            account_url=settings.storage_account_url,
            container_name=settings.blob_container,
            blob_name=document_path,
            credential=self.credential,
        ) as blob_client:
            return blob_client.download_blob().readall()



class ZavaDocumentParser(DocumentParser):
    """Parse PDF and Markdown documents using existing implementations."""

    def __init__(
        self,
        pdf_parser: Callable[[bytes], dict],
    ) -> None:
        self.pdf_parser = pdf_parser

    def parse_document(
        self,
        document: bytes,
        source_document: str,
    ) -> ParsedDocument:
        suffix = PurePosixPath(source_document).suffix.lower()

        if suffix == ".pdf":
            parsed = self.pdf_parser(document)

            return ParsedDocument(
                source_document=source_document,
                document_type="pdf",
                pages=parsed["pages"],
                figures=parsed["figures"],
            )

        if suffix in {".md", ".markdown"}:
            return ParsedDocument(
                source_document=source_document,
                document_type="markdown",
                text=document.decode("utf-8-sig"),
            )

        raise ValueError(
            f"Unsupported document type: {source_document}"
        )


class ZavaDocumentSplitter(DocumentSplitter):
    """Split parsed PDF and Markdown documents using existing functions."""

    def __init__(
        self,
        pdf_splitter: Callable,
        markdown_splitter: Callable,
    ) -> None:
        self.pdf_splitter = pdf_splitter
        self.markdown_splitter = markdown_splitter

    def split_document(
        self,
        parsed_data: ParsedDocument,
    ) -> list[dict]:
        if parsed_data.document_type == "pdf":
            return self.pdf_splitter(
                parsed_data.pages,
                parsed_data.figures,
                parsed_data.source_document,
            )

        if parsed_data.document_type == "markdown":
            return self.markdown_splitter(
                parsed_data.text.encode("utf-8"),
                parsed_data.source_document,
            )

        raise ValueError(
            f"Unsupported document type: {parsed_data.document_type}"
        )


class ZavaChunkEmbedder(ChunkEmbedder):
    """Generate embeddings using the existing cached implementation."""

    def embed_chunk(self, chunk: dict) -> list[float]:
        return get_chunk_embedding(chunk)


class Neo4jChunkLoader(ChunkLoader):
    """Store document chunks and embeddings in Neo4j."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def load_chunk(
        self,
        chunk: dict,
        embedding: list[float],
    ) -> None:
        store_chunk(self.session, chunk, embedding)
