import argparse

from graphrag.config import settings
from graphrag.neo4j_store import create_neo4j_driver


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check service configuration without displaying secret values."
    )
    parser.add_argument("--connect", action="store_true", help="Verify Neo4j connectivity.")
    args = parser.parse_args(argv)

    missing = False
    for label, setting_name in (
        ("Foundry", "foundry"),
        ("Azure OpenAI embeddings", "azure_openai"),
        ("Embedding deployment", "azure_openai_embedding_deployment"),
        ("Neo4j", "neo4j"),
    ):
        try:
            getattr(settings, setting_name)
        except RuntimeError as error:
            print(f"{label}: {error}")
            missing = True
        else:
            print(f"{label}: required settings present (values hidden).")
    if missing:
        return 1

    if args.connect:
        try:
            with create_neo4j_driver() as driver:
                driver.verify_connectivity()
        except Exception:
            print("Neo4j connection failed; verify local credentials and that the database is running.")
            return 1
        print("Neo4j connection verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())