# Quick Run — 30-Second Setup

For developers who just want to get it running.

## Prerequisites
- Python 3.12+
- Docker (for Neo4j)
- Azure CLI (`az login` done)
- Foundry credentials in `.env`

## Setup (One-Time)
```bash
# Install dependencies
uv sync

# Copy and configure environment files
cp .env.example .env
cp .env.database-mcp.example .env.database-mcp

# Edit .env with your Foundry project details
# Edit .env.database-mcp with Neo4j connection (or use defaults below)

# Start Neo4j
docker run -d --name neo4j-zava -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password neo4j:latest
```

## Run (In 3 Terminal Tabs)

**Tab 1 — Database MCP Server:**
```bash
cd src
uv run python -m database_mcp.server
# Should show: INFO: Uvicorn running on http://127.0.0.1:8002
```

**Tab 2 — Chatbot Backend:**
```bash
cd src
uv run python -m chatbot.chat_bot_main
# Should show: INFO: Uvicorn running on http://127.0.0.1:8000
```

**Tab 3 — Streamlit UI:**
```bash
cd src
uv run streamlit run UI/streamlit_chat.py
# Should open browser at http://localhost:8501
```

## Done! 🎉

- 💬 Chat at http://localhost:8501
- 📊 Neo4j admin at http://localhost:7474 (user: neo4j, pwd: password)
- 🔧 API at http://localhost:8000
- 📡 MCP server at http://localhost:8002

## Stop
```bash
# In each tab: Ctrl+C

# Stop Neo4j
docker stop neo4j-zava
```

## Troubleshooting
- **"Cannot reach chatbot"** → Check Terminal 2 is running
- **"Neo4j connection failed"** → Check `docker ps`, restart with the docker run command above
- **"Missing NEO4J_URI"** → Fill in `.env.database-mcp` with Neo4j credentials
- **"Missing FOUNDRY_PROJECT_ID"** → Fill in `.env` with your Foundry project details
