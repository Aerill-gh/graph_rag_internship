from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class ParsedDocument:
    """Document content and metadata passed from parser to splitter."""

    source_document: str
    document_type: str
    pages: list[dict] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    text: str = ""


@dataclass(frozen=True)
class DocumentSource:
    """Describe a document independently of where or how it is stored."""

    name: str
    location: object | None = None
    content: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DocumentSource requires a nonempty name.")
        if self.content is None and self.location is None:
            raise ValueError("DocumentSource requires content or a location.")
        if self.content is not None and not isinstance(self.content, bytes):
            raise ValueError("DocumentSource content must be bytes.")


@dataclass
class Chunk:
    """A source-linked text chunk that can hold its generated embedding."""

    chunk_id: str
    text: str
    source_document: str = ""
    pages: list[int] = field(default_factory=list)
    token_count: int = 0
    figures: list[dict] = field(default_factory=list)
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("Chunk requires a nonempty chunk_id.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Chunk requires nonempty text.")
        if self.source_document and not isinstance(self.source_document, str):
            raise ValueError("Chunk source_document must be a string.")
        if self.token_count < 0:
            raise ValueError("Chunk token_count cannot be negative.")

    @classmethod
    def from_mapping(
        cls,
        record: dict,
        source_document: str = "",
    ) -> "Chunk":
        if not isinstance(record, dict):
            raise TypeError("Chunk records must be mappings.")
        text = record.get("text")
        record_source = record.get("source_document")
        if not isinstance(record_source, str) or not record_source.strip():
            record_source = source_document if isinstance(source_document, str) else ""
        return cls(
            chunk_id=record.get("chunk_id", ""),
            text=text,
            source_document=record_source,
            pages=list(record.get("pages", [])),
            token_count=record.get("token_count", len(text.split()) if isinstance(text, str) else 0),
            figures=list(record.get("figures", [])),
            embedding=record.get("embedding"),
        )

    def set_embedding(self, embedding: list[float]) -> None:
        if not isinstance(embedding, list) or not embedding or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in embedding
        ):
            raise ValueError("Chunk embedding must be a nonempty list of finite numbers.")
        self.embedding = [float(value) for value in embedding]

    def to_record(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_document": self.source_document,
            "pages": self.pages,
            "text": self.text,
            "token_count": self.token_count,
            "figures": self.figures,
        }


class DocumentLoader(ABC):
    """Interface for loading documents from a data source."""

    @abstractmethod
    def load(self, source: str | DocumentSource) -> bytes:
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
    ) -> list[Chunk]:
        """Split parsed document content into chunks."""
        pass



class ChunkEmbedder(ABC):
    """Interface for generating chunk embeddings."""

    @abstractmethod
    def embed_chunk(self, chunk: Chunk) -> list[float]:
        pass


class ChunkLoader(ABC):
    """Interface for storing chunks and their embeddings."""

    @abstractmethod
    def load_chunk(
        self,
        chunk: Chunk,
    ) -> None:
        pass


class DocumentProcessingPipeline(ABC):
    """Common interface for document processing pipelines."""

    @abstractmethod
    def load_document(self, source: str | DocumentSource) -> None:
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

    def load_document(self, source: str | DocumentSource) -> None:
        if isinstance(source, DocumentSource):
            source_name = source.name
            if source.content is not None:
                document = source.content
            else:
                document = self.document_loader.load(source)
        else:
            source_name = source
            document = self.document_loader.load(source)

        parsed_data = self.document_parser.parse_document(
            document,
            source_name,
        )
        split_chunks = self.document_splitter.split_document(parsed_data)

        for split_chunk in split_chunks:
            chunk = (
                split_chunk
                if isinstance(split_chunk, Chunk)
                else Chunk.from_mapping(split_chunk, parsed_data.source_document)
            )
            embedding = self.chunk_embedder.embed_chunk(chunk)
            chunk.set_embedding(embedding)
            self.chunk_loader.load_chunk(chunk)
