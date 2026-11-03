"""Database-independent payloads exchanged by ingestion and MCP."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from graphrag.graph_extraction import validate_extraction
from graphrag.ontology import ALLOWED_RELATIONSHIPS, NODE_TYPES


EMBEDDING_DIMENSIONS = 1536
ENTITY_LABELS = NODE_TYPES - {"Document", "Chunk"}
DOMAIN_PATTERNS = sorted(
    pattern for pattern in ALLOWED_RELATIONSHIPS
    if pattern[1] not in {"HAS_CHUNK", "MENTIONS"}
)


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


class SearchDocumentsRequest(Contract):
    embedding: list[float] = Field(
        min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS,
    )
    top_k: int = Field(default=5, ge=1, le=100)


class SearchPassage(Contract):
    chunk_id: str = Field(min_length=1)
    source_document: str = Field(min_length=1)
    pages: list[int]
    text: str = Field(min_length=1)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    score: float


class SearchDocumentsResponse(Contract):
    passages: list[SearchPassage]


class FindEntitiesRequest(Contract):
    name: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("name")
    @classmethod
    def require_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Entity name cannot be blank.")
        return value


class EntityCandidate(Contract):
    labels: list[str]
    name: str = Field(min_length=1)


class FindEntitiesResponse(Contract):
    candidates: list[EntityCandidate]


class ExploreEntityRequest(Contract):
    name: str = Field(min_length=1, max_length=200)
    label: str
    limit: int = Field(default=20, ge=1, le=50)

    @field_validator("name")
    @classmethod
    def require_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Entity name cannot be blank.")
        return value

    @field_validator("label")
    @classmethod
    def require_allowed_label(cls, value: str) -> str:
        if value not in ENTITY_LABELS:
            raise ValueError("Unsupported entity label.")
        return value


class GraphFact(Contract):
    source: str = Field(min_length=1)
    relationship: str = Field(min_length=1)
    target: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document: str = Field(min_length=1)
    pages: list[int]
    text: str = Field(min_length=1)
    source_labels: list[str] = Field(default_factory=list)
    target_labels: list[str] = Field(default_factory=list)


class ExploreEntityResponse(Contract):
    facts: list[GraphFact]
    truncated: bool


class QueryGraphRequest(Contract):
    pattern: str
    source_names: list[str] = Field(min_length=1, max_length=5)
    target_name: str = Field(default="", max_length=200)
    match_mode: Literal["any", "all", "difference"] = "any"
    limit: int = Field(default=20, ge=1, le=50)

    @field_validator("pattern")
    @classmethod
    def require_allowed_pattern(cls, value: str) -> str:
        if value not in {"|".join(pattern) for pattern in DOMAIN_PATTERNS}:
            raise ValueError("Unsupported directed ontology pattern.")
        return value

    @field_validator("source_names")
    @classmethod
    def require_distinct_names(cls, value: list[str]) -> list[str]:
        if any(not name.strip() or len(name) > 200 for name in value):
            raise ValueError("Source names must be nonblank and at most 200 characters.")
        if len({name.strip().casefold() for name in value}) != len(value):
            raise ValueError("Source names must be distinct.")
        return value

    @model_validator(mode="after")
    def validate_match_mode(self) -> Self:
        if self.match_mode == "difference" and len(self.source_names) != 2:
            raise ValueError("Difference queries require exactly two source names.")
        return self


class QueryGraphResponse(Contract):
    facts: list[GraphFact]
    cypher: str
    parameters: dict[str, Any]
    result_limit: int
    truncated: bool
    scope: str


class CorpusSummaryResponse(Contract):
    database: str = Field(min_length=1)
    chunks: int = Field(ge=0)
    nodes: int = Field(ge=0)
    relationships: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
