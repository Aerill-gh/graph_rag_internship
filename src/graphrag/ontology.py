
# Zava GraphRAG Ontology

# Node types
NODE_TYPES = frozenset({
    "Company",
    "Document",
    "Chunk",
    "Employee",
    "Role",
    "Department",
    "Skill",
    "Policy",
    "ApplicabilityGroup",
    "Plan",
    "Service",
})

# Relationship types
RELATIONSHIP_TYPES = frozenset({
    "HAS_DEPARTMENT",
    "OFFERS",
    "HAS_CHUNK",
    "MENTIONS",
    "BELONGS_TO",
    "REQUIRES",
    "APPLIES_TO",
    "APPLIES_ON",
    "COVERS",
    "EXCLUDES",
    "HAS_ROLE",
})

# Allowed relationship patterns
# Format: (Source Node, Relationship, Target Node)
ALLOWED_RELATIONSHIPS = frozenset({
    ("Company", "HAS_DEPARTMENT", "Department"),
    ("Company", "OFFERS", "Plan"),
    ("Document", "HAS_CHUNK", "Chunk"),
    ("Chunk", "MENTIONS", "Role"),
    ("Chunk", "MENTIONS", "Policy"),
    ("Chunk", "MENTIONS", "Department"),
    ("Chunk", "MENTIONS", "Skill"),
    ("Chunk", "MENTIONS", "Plan"),
    ("Chunk", "MENTIONS", "Service"),
    ("Chunk", "MENTIONS", "Company"),
    ("Chunk", "MENTIONS", "Employee"),
    ("Chunk", "MENTIONS", "ApplicabilityGroup"),
    ("Role", "BELONGS_TO", "Department"),
    ("Role", "REQUIRES", "Skill"),
    ("Policy", "APPLIES_TO", "Role"),
    ("Policy", "APPLIES_TO", "ApplicabilityGroup"),
    ("Employee", "APPLIES_ON", "ApplicabilityGroup"),
    ("Employee", "HAS_ROLE", "Role"),
    ("Plan", "COVERS", "Service"),
    ("Plan", "EXCLUDES", "Service"),
})

# Display ontology information
if __name__ == "__main__":
    print("Node types:", len(NODE_TYPES))
    print("Relationship types:", len(RELATIONSHIP_TYPES))
    print("Allowed relationship patterns:", len(ALLOWED_RELATIONSHIPS))
