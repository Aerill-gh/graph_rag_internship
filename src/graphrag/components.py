from collections.abc import Callable
from pathlib import Path, PurePosixPath

from graphrag.database_mcp.client import DatabaseMcpClient
from graphrag.config import DOCUMENT_DIR, settings
from graphrag.embeddings import get_chunk_embedding
from graphrag.pipeline import (
    Chunk,
    ChunkEmbedder,
    ChunkLoader,
    DocumentSource,
    DocumentLoader,
    DocumentParser,
    DocumentSplitter,
    ParsedDocument,
)


class LocalDocumentLoader(DocumentLoader):
    """Load documents relative to local document storage."""

    def __init__(self, document_dir: Path = DOCUMENT_DIR) -> None:
        self.document_dir = document_dir.resolve()

    def load(self, source: str | DocumentSource) -> bytes:
        if isinstance(source, DocumentSource):
            if source.content is not None:
                return source.content
            location = source.location
        else:
            location = source
        if not isinstance(location, (str, Path)) or not str(location).strip():
            raise ValueError("Local document source requires a relative file path.")
        relative_path = Path(location)
        document_path = (self.document_dir / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not document_path.is_relative_to(self.document_dir)
        ):
            raise ValueError("The document must be inside local document storage.")
        return document_path.read_bytes()



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
    ) -> list[Chunk]:
        if parsed_data.document_type == "pdf":
            records = self.pdf_splitter(
                parsed_data.pages,
                parsed_data.figures,
                parsed_data.source_document,
            )
        elif parsed_data.document_type == "markdown":
            records = self.markdown_splitter(
                parsed_data.text.encode("utf-8"),
                parsed_data.source_document,
            )
        else:
            raise ValueError(
                f"Unsupported document type: {parsed_data.document_type}"
            )
        return [Chunk.from_mapping(record, parsed_data.source_document) for record in records]


class ZavaChunkEmbedder(ChunkEmbedder):
    """Generate embeddings using the existing cached implementation."""

    def embed_chunk(self, chunk: Chunk) -> list[float]:
        return get_chunk_embedding(chunk.to_record())


class McpChunkLoader(ChunkLoader):
    """Store document chunks and embeddings through the database MCP client."""

    def __init__(self, client: DatabaseMcpClient) -> None:
        self.client = client

    def load_chunk(
        self,
        chunk: Chunk,
    ) -> None:
        if chunk.embedding is None:
            raise ValueError("Cannot store a chunk before its embedding is set.")
        self.client.store_chunk(
            chunk.to_record(), chunk.embedding,
            embedding_model=settings.azure_openai_embedding_deployment,
        )
