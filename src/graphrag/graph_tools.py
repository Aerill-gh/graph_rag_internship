from graphrag.config import settings
from graphrag.neo4j_store import ENTITY_LABELS, create_neo4j_driver
from graphrag.ontology import ALLOWED_RELATIONSHIPS
from neo4j import unit_of_work


DOMAIN_PATTERNS = sorted(
    pattern for pattern in ALLOWED_RELATIONSHIPS
    if pattern[1] not in {"HAS_CHUNK", "MENTIONS"}
)


def _validate_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Limit must be an integer between 1 and 50.")


def find_entities(name: str, limit: int = 10) -> list[dict]:
    """Find ontology entities by name without embeddings or Azure calls."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A nonempty entity name is required.")
    _validate_limit(limit)
    with create_neo4j_driver() as driver:
        with driver.session(database=settings.neo4j.database) as session:
            return session.execute_read(lambda transaction: transaction.run(
                "MATCH (entity) WHERE any(label IN labels(entity) WHERE label IN $labels) "
                "AND toLower(entity.name) CONTAINS toLower($name) "
                "RETURN labels(entity) AS labels, entity.name AS name "
                "ORDER BY name LIMIT $limit",
                labels=sorted(ENTITY_LABELS), name=name.strip(), limit=limit,
            ).data())


def explore_entity(name: str, label: str, limit: int = 20) -> list[dict]:
    """Return source-backed domain relationships for an exact entity name and label."""
    if not isinstance(name, str) or not name.strip() or label not in ENTITY_LABELS:
        raise ValueError("An ontology label and nonempty entity name are required.")
    _validate_limit(limit)
    with create_neo4j_driver() as driver:
        with driver.session(database=settings.neo4j.database) as session:
            return session.execute_read(lambda transaction: transaction.run(
                "MATCH (source)-[fact]->(target) "
                "WHERE (source.name = $name AND $label IN labels(source)) "
                "OR (target.name = $name AND $label IN labels(target)) "
                "WITH source, fact, target WHERE size(coalesce(fact.source_chunk_ids, [])) > 0 "
                "UNWIND fact.source_chunk_ids AS chunk_id "
                "MATCH (chunk:Chunk {chunk_id: chunk_id}) "
                "RETURN labels(source) AS source_labels, source.name AS source, "
                "type(fact) AS relationship, labels(target) AS target_labels, target.name AS target, "
                "chunk.chunk_id AS chunk_id, chunk.source_document AS document, "
                "chunk.pages AS pages, chunk.text AS text "
                "ORDER BY source, relationship, target, chunk_id LIMIT $limit",
                name=name.strip(), label=label, limit=limit,
            ).data())


def query_graph(
    source_label: str,
    relationship: str,
    target_label: str,
    source_names: list[str],
    target_name: str = "",
    match_mode: str = "any",
    limit: int = 20,
) -> dict:
    """Compile an ontology-validated query plan into bounded, read-only Cypher."""
    _validate_limit(limit)
    if (source_label, relationship, target_label) not in DOMAIN_PATTERNS:
        raise ValueError("Unsupported directed ontology pattern.")
    if (
        not isinstance(source_names, list) or not 1 <= len(source_names) <= 5
        or any(not isinstance(name, str) or not name.strip() or len(name) > 200 for name in source_names)
        or not isinstance(target_name, str) or len(target_name) > 200
        or match_mode not in {"any", "all", "difference"}
    ):
        raise ValueError("Invalid graph query filters.")
    if match_mode == "difference" and len(source_names) != 2:
        raise ValueError("Difference queries require exactly two source names.")
    names = [name.strip() for name in source_names]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Source names must be distinct.")
    conditions = [
        "($target_name = '' OR toLower(target.name) CONTAINS toLower($target_name))",
        "size(coalesce(fact.source_chunk_ids, [])) > 0",
    ]
    if match_mode == "difference":
        conditions.append("toLower(source.name) CONTAINS toLower($source_names[0])")
        conditions.append(
            f"NOT EXISTS {{ MATCH (other:{source_label})-[:{relationship}]->(target) "
            "WHERE toLower(other.name) CONTAINS toLower($source_names[1]) }"
        )
    else:
        conditions.append(
            "any(name IN $source_names WHERE toLower(source.name) CONTAINS toLower(name))"
        )
        if match_mode == "all":
            conditions.append(
                "all(name IN $source_names WHERE EXISTS { "
                f"MATCH (other:{source_label})-[other_fact:{relationship}]->(target) "
                "WHERE toLower(other.name) CONTAINS toLower(name) "
                "AND size(coalesce(other_fact.source_chunk_ids, [])) > 0 })"
            )
    cypher = (
        f"MATCH (source:{source_label})-[fact:{relationship}]->(target:{target_label}) "
        "WHERE " + " AND ".join(conditions) + " "
        "UNWIND fact.source_chunk_ids AS chunk_id "
        "MATCH (chunk:Chunk {chunk_id: chunk_id}) "
        "RETURN source.name AS source, type(fact) AS relationship, target.name AS target, "
        "chunk.chunk_id AS chunk_id, chunk.source_document AS document, "
        "chunk.pages AS pages, chunk.text AS text "
        "ORDER BY source, target, chunk_id LIMIT $limit"
    )
    parameters = {"source_names": names, "target_name": target_name.strip(), "limit": limit + 1}
    @unit_of_work(timeout=10)
    def read(transaction):
        return transaction.run(cypher, **parameters).data()

    with create_neo4j_driver() as driver:
        with driver.session(database=settings.neo4j.database) as session:
            rows = session.execute_read(read)
    return {
        "facts": rows[:limit], "cypher": cypher,
        "parameters": parameters, "result_limit": limit,
        "truncated": len(rows) > limit,
        "scope": "Extracted graph facts only; absence is not evidence of exclusion."
    }