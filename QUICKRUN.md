# Quick Run

Run the local Zava GraphRAG system on Windows with PowerShell.

## Prerequisites

- Python 3.12+ and `uv`
- Neo4j Desktop 2 with the GraphRAG database in a local DBMS
- Access to the Foundry project and Azure sign-in for chat and ingestion

## One-Time Setup

From the repository root:

```powershell
uv sync
if (!(Test-Path .env)) { Copy-Item .env.example .env }
if (!(Test-Path .env.database-mcp)) { Copy-Item .env.database-mcp.example .env.database-mcp }
az login
```

Fill in the Foundry and Azure values in `.env`. In `.env.database-mcp`, set the Neo4j URI, username, password, and `NEO4J_DATABASE` to the name of the existing database that contains your GraphRAG data. The database name can differ between users; it must match a database in the selected Neo4j Desktop DBMS.

For running the UI in a separate terminal, set these in `.env`:

```dotenv
BACKEND_API_URL=http://127.0.0.1:8000
CHATBOT_API_URL=http://127.0.0.1:8000/chat
```

Do not put `$env:` PowerShell syntax in `.env`.

## Recommended: Three Terminals

Open three PowerShell terminals at the repository root. Start them in this order and leave each terminal running.

**Terminal 1 - Neo4j, MCP, and GraphRAG ingestion API:**

```powershell
uv run python -m graphrag.main
```

This starts the Neo4j Desktop DBMS if the configured database is not already reachable, then starts the Database MCP server and ingestion API. The MCP default is `http://127.0.0.1:8002/mcp`. The GraphRAG API uses `GRAPHRAG_PORT` from `.env` and defaults to port `8001`.

**Terminal 2 - Chatbot API:**

```powershell
uv run python -m chatbot.main
```

The default address is `http://127.0.0.1:8000`.

**Terminal 3 - Streamlit UI:**

```powershell
uv run python -m ui.main
```

Open `http://127.0.0.1:8501`.

The Neo4j Desktop DBMS remains running after the GraphRAG services stop. Stop it from Neo4j Desktop when it is no longer needed. When started outside the Desktop UI, its card may still display `STOPPED`; do not press Play while Bolt port `7687` is already in use.

## Verify

Use the GraphRAG port configured in `.env` (`8001` by default):

```powershell
Invoke-RestMethod http://127.0.0.1:8001/documents
```

Check the chatbot API at `http://127.0.0.1:8000/docs`. To try a chat request, use the UI or run:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -ContentType "application/json" `
  -Body '{"message":"What does PerksPlus cover?","mode":"baseline","history":[]}'
```

Chat and ingestion requests call Azure services and may incur costs. Ingestion can be tested from the GraphRAG API docs at `/docs` with a document in `document_storage/`.

## Alternative: Root Launcher

If Neo4j is already running in Neo4j Desktop, the repository-root launcher starts MCP, GraphRAG API, chatbot API, and UI together:

```powershell
uv run python main.py
```

Open `http://127.0.0.1:8501`. This launcher does not start Neo4j. Use this alternative instead of the three-terminal sequence; do not run both because they use the same service ports.
