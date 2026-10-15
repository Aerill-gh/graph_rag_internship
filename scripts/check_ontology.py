from src.graphrag.ontology import (
    NODE_TYPES,
    RELATIONSHIP_TYPES,
    ALLOWED_RELATIONSHIPS,
)

print("Zava GraphRAG Ontology")
print("Node types:", len(NODE_TYPES))
print("Relationship types:", len(RELATIONSHIP_TYPES))
print("Allowed patterns:", len(ALLOWED_RELATIONSHIPS))

for source, relation, target in sorted(ALLOWED_RELATIONSHIPS):
    print(f"{source} -> {relation} -> {target}")