"""Load the MCP server's database configuration without exporting it to clients."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatabaseSettings:
    uri: str
    username: str
    password: str = field(repr=False)
    database: str = "neo4j"


def load_database_settings() -> DatabaseSettings:
    values = dotenv_values(PROJECT_ROOT / ".env.database-mcp")

    def required(name: str) -> str:
        value = values.get(name)
        if not value:
            raise RuntimeError(
                f"Missing {name}. Set it in .env.database-mcp for the database MCP server."
            )
        return value

    return DatabaseSettings(
        uri=required("NEO4J_URI"),
        username=required("NEO4J_USERNAME"),
        password=required("NEO4J_PASSWORD"),
        database=values.get("NEO4J_DATABASE") or "neo4j",
    )


def database_mcp_url() -> str:
    # Load only the client's endpoint; never load the server's credentials.
    values = dotenv_values(PROJECT_ROOT / ".env")
    value = os.getenv("DATABASE_MCP_URL", values.get("DATABASE_MCP_URL") or "")
    url = value.strip() or "http://127.0.0.1:8002/mcp"
    if not url.startswith(("http://", "https://")):
        raise ValueError("DATABASE_MCP_URL must be an HTTP or HTTPS MCP endpoint.")
    return url

