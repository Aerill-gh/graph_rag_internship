
from ontology import NODE_TYPES, ALLOWED_RELATIONSHIPS


def validate_node(node_type):
    return node_type in NODE_TYPES


def validate_relationship(source, relationship, target):
    return (source, relationship, target) in ALLOWED_RELATIONSHIPS


# Test nodes
print("Employee:", validate_node("Employee"))
print("Role:", validate_node("Role"))

# Test relationships
print("Role -> Skill:", validate_relationship(
    "Role", "REQUIRES", "Skill"
))

print("Skill -> Role:", validate_relationship(
    "Skill", "REQUIRES", "Role"
))

print("Document -> Chunk:", validate_relationship(
    "Document", "HAS_CHUNK", "Chunk"
))
print("Chunk -> Role:", validate_relationship(
    "Chunk", "MENTIONS", "Role"
))