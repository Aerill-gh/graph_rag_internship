"""Database-independent payloads exchanged by ingestion and MCP."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from graphrag.graph_extraction import validate_extraction


EMBEDDING_DIMENSIONS = 1536


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class ChunkPayload(Contract):
    chunk_id: str = Field(min_length=1)
    source_document: str = Field(min_length=1)
    text: str = Field(min_length=1)
    pages: list[int]
    token_count: int = Field(ge=0)
    figures: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("chunk_id", "source_document", "text")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chunk identifiers, source and text cannot be blank.")
        return value

    @field_validator("pages")
    @classmethod
    def require_positive_pages(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value):
            raise ValueError("Page numbers must be positive integers.")
        return value


class ChunkWrite(Contract):
    chunk: ChunkPayload
    embedding: list[float] | None = Field(
        default=None, min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS,
    )
    embedding_model: str | None = None

    @model_validator(mode="after")
    def require_embedding_model(self) -> Self:
        if self.embedding is not None:
            if self.embedding_model is None or not self.embedding_model.strip():
                raise ValueError("An embedding model is required with an embedding.")
        elif self.embedding_model is not None:
            raise ValueError("An embedding model requires an embedding.")
        return self


class Entity(Contract):
    label: str
    name: str


class Relationship(Contract):
    source_label: str
    source_name: str
    type: str
    target_label: str
    target_name: str


class ExtractionPayload(Contract):
    entities: list[Entity]
    relationships: list[Relationship]

    @model_validator(mode="after")
    def require_allowed_facts(self) -> Self:
        validate_extraction(self.model_dump())
        return self


class ExtractionWrite(Contract):
    chunk_id: str = Field(min_length=1)
    extraction: ExtractionPayload

    @field_validator("chunk_id")
    @classmethod
    def require_chunk_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source chunk ID cannot be blank.")
        return value


class HealthResponse(Contract):
    status: Literal["ok"] = "ok"


class SchemaResponse(Contract):
    status: Literal["ready"] = "ready"


class ChunkWriteResponse(Contract):
    chunk_id: str = Field(min_length=1)


class ExtractionWriteResponse(Contract):
    chunk_id: str = Field(min_length=1)
    entity_mentions: int = Field(ge=0)
    relationship_mentions: int = Field(ge=0)


class ChunkReadResponse(Contract):
    chunk: ChunkPayload
    embedding: list[float] | None = Field(
        min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS,
    )
    embedding_model: str | None
    entities: list[Entity]
    relationships: list[Relationship]
    entities_truncated: bool
    relationships_truncated: bool
