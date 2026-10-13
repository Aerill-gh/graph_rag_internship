
# Zava GraphRAG Ontology

# Node types
NODE_TYPES = frozenset({
    "Company",
    "Department",
    "Employee",
    "Role",
    "Skill",
    "Policy",
    "Plan",
    "Service",
    "Document",
    "Chunk",
})

# Relationship types
RELATIONSHIP_TYPES = frozenset({
    "HAS_DEPARTMENT",
    "OFFERS",
    "COVERS",
    "EXCLUDES",
    "HAS_ROLE",
    "BELONGS_TO",
    "REQUIRES",
    "HAS_CHUNK",
    "MENTIONS",
    "APPLIES_TO",
})

# Allowed relationship patterns
# Format: (Source Node, Relationship, Target Node)
ALLOWED_RELATIONSHIPS = frozenset({
    ("Company", "HAS_DEPARTMENT", "Department"),
    ("Company", "OFFERS", "Plan"),
    ("Role", "BELONGS_TO", "Department"),
    ("Role", "REQUIRES", "Skill"),
    ("Employee", "HAS_ROLE", "Role"),
    ("Policy", "APPLIES_TO", "Role"),
    ("Plan", "COVERS", "Service"),
    ("Plan", "EXCLUDES", "Service"),
    ("Document", "HAS_CHUNK", "Chunk"),

    # Chunk-to-entity relationships
    ("Chunk", "MENTIONS", "Role"),
    ("Chunk", "MENTIONS", "Policy"),
    ("Chunk", "MENTIONS", "Plan"),
    ("Chunk", "MENTIONS", "Department"),
    ("Chunk", "MENTIONS", "Skill"),
    ("Chunk", "MENTIONS", "Service"),
})

# Display ontology information
if __name__ == "__main__":
    print("Node types:", len(NODE_TYPES))
    print("Relationship types:", len(RELATIONSHIP_TYPES))
    print("Allowed relationship patterns:", len(ALLOWED_RELATIONSHIPS))
