from graphrag.database_mcp.client import DatabaseMcpClient
from graphrag.database_mcp.contracts import (
    DOMAIN_PATTERNS,
    ENTITY_LABELS,
    ExploreEntityRequest,
    FindEntitiesRequest,
    QueryGraphRequest,
)


def _validate_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Limit must be an integer between 1 and 50.")


def find_entities(name: str, limit: int = 10) -> list[dict]:
    request = FindEntitiesRequest(name=name, limit=limit)
    with DatabaseMcpClient() as client:
        result = client.find_entities(request.name.strip(), request.limit)
    return [candidate.model_dump() for candidate in result.candidates]


def explore_entity(name: str, label: str, limit: int = 20) -> list[dict]:
    """Return source-backed domain relationships for an exact entity name and label."""
    request = ExploreEntityRequest(name=name, label=label, limit=limit)
    with DatabaseMcpClient() as client:
        result = client.explore_entity(request.name.strip(), request.label, request.limit)
    return [fact.model_dump() for fact in result.facts]


def query_graph(
    source_label: str,
    relationship: str,
    target_label: str,
    source_names: list[str],
    target_name: str = "",
    match_mode: str = "any",
    limit: int = 20,
) -> dict:
    """Query bounded, ontology-validated graph facts through the Database MCP service."""
    pattern = (source_label, relationship, target_label)
    if pattern not in DOMAIN_PATTERNS:
        raise ValueError("Unsupported directed ontology pattern.")
    request = QueryGraphRequest(
        pattern="|".join(pattern), source_names=source_names, target_name=target_name,
        match_mode=match_mode, limit=limit,
    )
    with DatabaseMcpClient() as client:
        result = client.query_graph(
            request.pattern, source_names=request.source_names, target_name=request.target_name,
            match_mode=request.match_mode, limit=request.limit,
        )
    return result.model_dump()