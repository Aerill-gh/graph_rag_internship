from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any



@dataclass
class ParsedDocument:
    """Document content and metadata passed from parser to splitter."""

    source_document: str
    document_type: str
    pages: list[dict] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    text: str = ""


class DocumentLoader(ABC):
    """Interface for loading documents from a data source."""

    @abstractmethod
    def load(self, document_path: str) -> bytes:
        """Load and return the document's bytes."""
        pass



class DocumentParser(ABC):
    """Interface for parsing documents."""

    @abstractmethod
    def parse_document(
        self,
        document: bytes,
        source_document: str,
    ) -> ParsedDocument:
        """Parse document content while preserving its source."""
        pass



class DocumentSplitter(ABC):
    """Interface for splitting parsed documents into chunks."""

    @abstractmethod
    def split_document(
        self,
        parsed_data: ParsedDocument,
    ) -> list[dict]:
        """Split parsed document content into chunks."""
        pass



class ChunkEmbedder(ABC):
    """Interface for generating chunk embeddings."""

    @abstractmethod
    def embed_chunk(self, chunk: dict) -> list[float]:
        pass


class ChunkLoader(ABC):
    """Interface for storing chunks and their embeddings."""

    @abstractmethod
    def load_chunk(
        self,
        chunk: dict,
        embedding: list[float],
    ) -> None:
        pass


class DocumentProcessingPipeline(ABC):
    """Common interface for document processing pipelines."""

    @abstractmethod
    def load_document(self, path_to_document: str) -> None:
        pass



class CompositeDocumentProcessingPipeline(DocumentProcessingPipeline):
    """Coordinate document loading, parsing, chunking, embedding, and storage."""

    def __init__(
        self,
        document_loader: DocumentLoader,
        document_parser: DocumentParser,
        document_splitter: DocumentSplitter,
        chunk_embedder: ChunkEmbedder,
        chunk_loader: ChunkLoader,
    ) -> None:
        self.document_loader = document_loader
        self.document_parser = document_parser
        self.document_splitter = document_splitter
        self.chunk_embedder = chunk_embedder
        self.chunk_loader = chunk_loader

    def load_document(self, path_to_document: str) -> None:
        document = self.document_loader.load(path_to_document)

        parsed_data = self.document_parser.parse_document(
            document,
            path_to_document,
        )
        chunks = self.document_splitter.split_document(parsed_data)

        for chunk in chunks:
            embedding = self.chunk_embedder.embed_chunk(chunk)
            self.chunk_loader.load_chunk(chunk, embedding)
