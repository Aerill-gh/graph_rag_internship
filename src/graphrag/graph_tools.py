from graphrag.config import settings
from graphrag.neo4j_store import ENTITY_LABELS, create_neo4j_driver


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