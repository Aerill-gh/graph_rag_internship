from typing import TypedDict

from graphrag.database_mcp.client import DatabaseMcpClient
from graphrag.database_mcp.contracts import DOMAIN_PATTERNS, ENTITY_LABELS
from graphrag.models import generate_embedding


TOP_K = 3


SEARCH_DOCUMENTS_DESCRIPTION = (
    "Search the indexed Zava document chunks for evidence relevant to a question. "
    "Returns ranked source passages with chunk ID, document, pages, and similarity. "
    "Use the returned passages as evidence and cite their source. Results may not be exhaustive."
)

EXPLORE_ENTITY_DESCRIPTION = (
    "Look up a named Zava graph entity and its source-backed neighboring facts. "
    "Use for a named plan, role, policy, department, or skill. Ambiguous names return "
    "candidates; retry using an exact candidate name. Facts are extracted, not independently verified."
)

QUERY_GRAPH_DESCRIPTION = (
    "Query directed graph relationships for lists, shared skills, and comparisons. "
    "Choose an allowed pattern and source-name filters; use any for combined facts, "
    "all for targets shared by every named source, difference for targets linked to "
    "the first source but not the second. Names are case-insensitive substring filters. "
    "Returns bounded facts, source passages, executed Cypher, and truncation status. "
    "Missing edges are NOT evidence of exclusion; confirm coverage and conditions in passages."
)


class SearchResult(TypedDict):
    chunk_id: str
    document: str
    pages: list[int]
    text: str
    score: float


def graph_tool_parameters() -> dict[str, dict]:
    return {
        "explore_entity": {
            "name": {"type": "string"},
            "label": {"type": "string", "enum": sorted(ENTITY_LABELS)},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "query_graph": {
            "pattern": {"type": "string", "enum": ["|".join(pattern) for pattern in DOMAIN_PATTERNS]},
            "source_names": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
            "target_name": {"type": "string", "description": "Target substring, or empty string for all targets."},
            "match_mode": {"type": "string", "enum": ["any", "all", "difference"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
    }


def _graph_passages(rows: list[dict]) -> list[dict]:
    passages = {}
    for row in rows:
        key = (row["document"], row["chunk_id"])
        fact = {field: row[field] for field in ("source", "relationship", "target")}
        if key not in passages:
            passages[key] = {
                "chunk_id": row["chunk_id"], "source_document": row["document"],
                "pages": row["pages"], "text": row["text"],
                "graph_fact": fact, "graph_facts": [],
            }
        if fact not in passages[key]["graph_facts"]:
            passages[key]["graph_facts"].append(fact)
    return list(passages.values())


def search_documents(question: str, top_k: int = 5) -> list[dict]:
    """Find Zava document passages relevant to a question.

    Return matching text, document names, page numbers, chunk IDs,
    and similarity scores so answers can cite their sources.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100.")

    embedding = generate_embedding(question)
    if not embedding:
        raise ValueError("Question embedding must not be empty.")

    with DatabaseMcpClient() as client:
        response = client.search_documents(embedding, top_k)
    return [passage.model_dump() for passage in response.passages]


def run_graph_tool(name: str, arguments: dict) -> dict:
    definitions = graph_tool_parameters()
    if name not in definitions:
        raise ValueError("Unsupported graph tool.")
    if not isinstance(arguments, dict) or set(arguments) != set(definitions[name]):
        raise ValueError("Graph tool arguments must exactly match the tool schema.")
    if name == "explore_entity":
        entity_name, label, limit = arguments["name"], arguments["label"], arguments["limit"]
        if label not in ENTITY_LABELS or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Invalid entity label or result limit.")
        with DatabaseMcpClient() as client:
            candidates = [
                candidate.model_dump()
                for candidate in client.find_entities(entity_name).candidates
                if label in candidate.labels
            ]
            exact = [candidate for candidate in candidates
                     if candidate["name"].casefold() == entity_name.strip().casefold()]
            selected = exact or candidates
            if len(selected) != 1:
                return {"passages": [], "candidates": candidates, "ambiguous": len(selected) > 1}
            response = client.explore_entity(selected[0]["name"], label, limit)
        return {
            "passages": _graph_passages([fact.model_dump() for fact in response.facts]),
            "potentially_truncated": response.truncated,
        }
    pattern = arguments["pattern"]
    allowed = {"|".join(pattern): pattern for pattern in DOMAIN_PATTERNS}
    if not isinstance(pattern, str) or pattern not in allowed:
        raise ValueError("Unsupported directed graph pattern.")
    with DatabaseMcpClient() as client:
        result = client.query_graph(
            pattern, source_names=arguments["source_names"],
            target_name=arguments["target_name"], match_mode=arguments["match_mode"],
            limit=arguments["limit"],
        )
    return {
        "passages": _graph_passages([fact.model_dump() for fact in result.facts]),
        "cypher": result.cypher,
        "parameters": result.parameters,
        "result_limit": result.result_limit,
        "truncated": result.truncated,
        "scope": result.scope,
    }
