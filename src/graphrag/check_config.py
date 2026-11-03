import argparse

from graphrag.config import settings
from database_mcp.client import DatabaseMcpClient
from database_mcp.config import database_mcp_url


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check service configuration without displaying secret values."
    )
    parser.add_argument("--connect", action="store_true", help="Verify database MCP connectivity.")
    args = parser.parse_args(argv)

    missing = False
    for label, setting_name in (
        ("Foundry", "foundry"),
        ("Azure OpenAI embeddings", "azure_openai"),
        ("Embedding deployment", "azure_openai_embedding_deployment"),
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

    try:
        database_mcp_url()
    except ValueError as error:
        print(f"Database MCP: {error}")
        return 1
    print("Database MCP: endpoint configured (value hidden).")

    if args.connect:
        try:
            with DatabaseMcpClient() as client:
                client.health_check()
        except Exception as error:
            print(
                f"Database MCP connection failed ({type(error).__name__}); "
                "check the MCP endpoint, server configuration and database availability."
            )
            return 1
        print("Database MCP connection verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())