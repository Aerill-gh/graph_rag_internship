import json

from graphrag.config import settings
from graphrag.embeddings import generate_embedding
from graphrag.neo4j_store import VECTOR_INDEX_NAME, create_neo4j_driver
from graphrag.graph_tools import DOMAIN_PATTERNS, explore_entity, find_entities, query_graph
from graphrag.neo4j_store import ENTITY_LABELS


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
        candidates = [candidate for candidate in find_entities(entity_name)
                      if label in candidate["labels"]]
        exact = [candidate for candidate in candidates
                 if candidate["name"].casefold() == entity_name.strip().casefold()]
        selected = exact or candidates
        if len(selected) != 1:
            return {"passages": [], "candidates": candidates, "ambiguous": len(selected) > 1}
        rows = explore_entity(selected[0]["name"], label, limit)
        return {"passages": _graph_passages(rows), "potentially_truncated": len(rows) == limit}
    pattern = arguments["pattern"]
    allowed = {"|".join(pattern): pattern for pattern in DOMAIN_PATTERNS}
    if not isinstance(pattern, str) or pattern not in allowed:
        raise ValueError("Unsupported directed graph pattern.")
    result = query_graph(
        *allowed[pattern], source_names=arguments["source_names"],
        target_name=arguments["target_name"], match_mode=arguments["match_mode"],
        limit=arguments["limit"],
    )
    return {"passages": _graph_passages(result.pop("facts")), **result}


SEARCH_CHUNKS_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node AS chunk, score
RETURN chunk.chunk_id AS chunk_id,
       chunk.source_document AS source_document,
       chunk.pages AS pages,
       chunk.text AS text,
       chunk.figures_json AS figures_json,
       score
ORDER BY score DESC
"""


def search_documents(question: str, top_k: int = 5) -> list[dict]:
    """Embed a question and return source-linked chunks from Neo4j."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string.")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100.")

    embedding = generate_embedding(question)
    if not embedding:
        raise ValueError("Question embedding must not be empty.")

    with create_neo4j_driver() as driver:
        with driver.session(database=settings.neo4j.database) as session:
            rows = session.run(
                SEARCH_CHUNKS_QUERY,
                index_name=VECTOR_INDEX_NAME,
                top_k=top_k,
                embedding=embedding,
            ).data()

    passages = []
    for row in rows:
        passage = dict(row)
        passage["figures"] = json.loads(passage.pop("figures_json", None) or "[]")
        passages.append(passage)
    return passages