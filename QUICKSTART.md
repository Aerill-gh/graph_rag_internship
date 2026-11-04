# Zava GraphRAG Agent — Quick Start Guide

Complete step-by-step instructions to run the entire system locally.

---

## 📋 Prerequisites

Before starting, make sure you have:

1. **Python 3.12+** installed
   ```bash
   python --version
   ```

2. **uv** package manager installed
   ```bash
   pip install uv
   ```

3. **Docker** installed and running (for Neo4j)
   ```bash
   docker --version
   ```

4. **Git** for cloning the repo
   ```bash
   git --version
   ```

5. **Azure CLI** (for authentication)
   ```bash
   az login
   ```

6. **Microsoft Foundry project** credentials configured
   - You should have access to an Azure AI Foundry project
   - Have model deployments for chat and embeddings
   - See `.env.example` for what's needed

---

## 🔧 Setup (One-Time)

### 1. Clone the Repository
```bash
git clone https://github.com/Aerill-gh/graph_rag_internship.git
cd graph_rag_internship
```

### 2. Install Dependencies
```bash
uv sync
```

This installs all Python packages from `pyproject.toml` into a virtual environment.

### 3. Configure Environment Variables

Copy the example files and fill in your credentials:

#### For Foundry and Azure services (`.env`):
```bash
cp .env.example .env
```

Then edit `.env` and add:
```
# Azure authentication
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>

# Foundry project
FOUNDRY_PROJECT_ID=<your-foundry-project-id>
FOUNDRY_REGION=<your-foundry-region>  # e.g., eastus

# Model deployments (in your Foundry project)
FOUNDRY_CHAT_DEPLOYMENT=<your-chat-model-name>
FOUNDRY_EMBEDDING_DEPLOYMENT=<your-embedding-model-name>
```

#### For Database MCP Server (`.env.database-mcp`):
```bash
cp .env.database-mcp.example .env.database-mcp
```

Then edit `.env.database-mcp` and add:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### 4. Start Neo4j Database

Start Neo4j in Docker:

```bash
docker run -d \
  --name neo4j-zava \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

Verify it's running:
```bash
docker logs neo4j-zava
```

You should see: `Started`.

Access Neo4j Browser at: http://localhost:7474
- Username: `neo4j`
- Password: `password`

### 5. Download Sample Data (Optional, for graph building)

```bash
uv run python scripts/download_data.py
```

This downloads the 7 sample Zava documents to `data/raw/`.

### 6. Build the Knowledge Graph (Optional, if you want to run the full pipeline)

This takes a few minutes and costs some Azure API tokens:

```bash
uv run python scripts/build_graph.py
```

This:
1. Reads PDFs with Document Intelligence
2. Chunks the documents
3. Generates embeddings
4. Extracts entities and relationships
5. Loads everything into Neo4j

Results are cached in `data/cache/`, so re-runs are cheaper.

---

## 🚀 Running the System

The system has **3 independent services** that run in separate terminals:

### Terminal 1: Start the Database MCP Server

```bash
cd src
uv run python -m database_mcp.server
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8002 (Press CTRL+C to quit)
```

**This server:**
- Wraps Neo4j behind the Model Context Protocol (MCP)
- Provides tools: `search_documents`, `explore_entity`, `query_graph`
- The chatbot calls this for all database operations

**Keep this terminal open.**

---

### Terminal 2: Start the Chatbot Backend API

```bash
cd src
uv run python -m chatbot.chat_bot_main
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**This server:**
- Provides REST API endpoints:
  - `POST /chat` - Answer a question
  - `GET /conversations` - List chats
  - `POST /conversations` - Create a new chat
  - `GET /conversations/{id}/messages` - Get chat history
  - `PATCH /conversations/{id}` - Update chat
  - `DELETE /conversations/{id}` - Delete chat
  - `GET /languages` - Available languages
- Manages chat history in SQLite (`data/history.sqlite3`)
- Calls the Database MCP Server for all database queries

**Keep this terminal open.**

---

### Terminal 3: Start the Streamlit UI

```bash
cd src
uv run streamlit run UI/streamlit_chat.py
```

You should see:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**The UI:**
- Chat interface for asking questions
- Shows answers with sources
- Manages conversations
- Calls the Chatbot Backend API

Open http://localhost:8501 in your browser.

---

## 📝 Usage Example

### 1. Ask a Question

In the Streamlit UI:
1. Type: "What are the qualifications for the CFO role?"
2. Select agent mode: "Baseline" or "GraphRAG"
3. Press Enter

### 2. View the Answer

The UI shows:
- The agent's answer
- Sources cited (document name, pages, chunks)
- Tools used (search_documents, explore_entity, etc.)

### 3. Save Conversations

Conversations are automatically saved when you ask questions:
- Create new chat with "New chat" button
- Rename with "Rename" button
- Pin/unpin to keep favorites visible
- Select language (English, Deutsch, Latviešu, Español)
- All stored in `data/history.sqlite3`

---

## 🧪 Testing

### Run All Tests

```bash
uv run python -m unittest discover tests -v
```

### Run Chatbot Tests Only

```bash
uv run python -m unittest tests.chatbot.test_chat_api -v
```

All tests should pass (8/8).

### Test Individual Services

**Test Neo4j connection:**
```bash
uv run python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
driver.verify_connectivity()
print('Neo4j connected!')
"
```

**Test Database MCP Server:**
```bash
curl http://127.0.0.1:8002/mcp
```

**Test Chatbot API:**
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the company about?",
    "mode": "baseline",
    "language": "en"
  }'
```

**Test Languages API:**
```bash
curl http://127.0.0.1:8000/languages
```

---

## 🛑 Stopping Everything

### Stop Streamlit (Terminal 3)
```
Press Ctrl+C
```

### Stop Chatbot Backend (Terminal 2)
```
Press Ctrl+C
```

### Stop Database MCP Server (Terminal 1)
```
Press Ctrl+C
```

### Stop Neo4j
```bash
docker stop neo4j-zava
```

### Remove Neo4j Container (to start fresh)
```bash
docker rm neo4j-zava
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'graphrag'"

**Cause:** Virtual environment not activated or dependencies not installed.

**Fix:**
```bash
uv sync
```

---

### "Cannot connect to Neo4j"

**Cause:** Neo4j container not running or wrong credentials.

**Check:**
```bash
docker ps | grep neo4j
```

**Restart:**
```bash
docker restart neo4j-zava
```

**Or restart clean:**
```bash
docker stop neo4j-zava
docker rm neo4j-zava
# Then run the docker run command from Setup step 4
```

---

### "RuntimeError: Missing NEO4J_URI"

**Cause:** `.env.database-mcp` not configured.

**Fix:**
```bash
cp .env.database-mcp.example .env.database-mcp
# Then edit and add NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
```

---

### "HTTP 503 - Agent unavailable"

**Cause:** Database MCP Server not running or Foundry credentials missing.

**Check:**
1. Is Database MCP Server running? (Terminal 1 should show "Uvicorn running on...")
2. Are Foundry credentials in `.env`? (AZURE_SUBSCRIPTION_ID, FOUNDRY_PROJECT_ID, etc.)
3. Is Neo4j running? (Check `docker ps`)

**Fix:** Ensure all 3 terminals are running and `.env` is configured.

---

### "Cannot reach the chatbot service"

**Cause:** Chatbot Backend not running or URL misconfigured.

**Check:**
1. Is Chatbot running? (Terminal 2 should show "Uvicorn running on...")
2. Is URL correct in UI? (Should be http://127.0.0.1:8000)

**Edit in UI code** (`src/UI/streamlit_chat.py`):
```python
DEFAULT_BACKEND_API_URL = "http://127.0.0.1:8000"
```

Or set in `.env`:
```
BACKEND_API_URL=http://127.0.0.1:8000
CHATBOT_API_URL=http://127.0.0.1:8000/chat
```

---

### "AttributeError: module 'graphrag' has no attribute..."

**Cause:** Cached Python bytecode is stale.

**Fix:**
```bash
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
uv sync
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────┐
│  Streamlit UI (http://localhost:8501)  │
│  ✅ Chat interface                      │
│  ✅ Conversation management             │
└────────────────┬────────────────────────┘
                 │ HTTP REST API
                 │ (JSON)
                 ▼
┌─────────────────────────────────────────┐
│  Chatbot Backend (http://localhost:8000)│
│  ✅ FastAPI server                      │
│  ✅ /chat endpoint                      │
│  ✅ /conversations endpoint             │
│  ✅ /languages endpoint                 │
│  ✅ Chat history (SQLite)              │
└────────────────┬────────────────────────┘
                 │ JSON-RPC MCP Protocol
                 │ (HTTP)
                 ▼
┌─────────────────────────────────────────┐
│  Database MCP Server (localhost:8002)   │
│  ✅ Model Context Protocol wrapper      │
│  ✅ Tools: search_documents             │
│  ✅ Tools: explore_entity               │
│  ✅ Tools: query_graph                  │
└────────────────┬────────────────────────┘
                 │ Bolt Protocol
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Neo4j Database (localhost:7687)        │
│  ✅ Knowledge graph                     │
│  ✅ Vector index                        │
│  ✅ Entities & relationships            │
└─────────────────────────────────────────┘

Separate components (Azure):
┌─────────────────────────────────────────┐
│  Microsoft Foundry (Azure)              │
│  ✅ Chat model deployment               │
│  ✅ Embedding model deployment          │
│  ✅ Azure AI Document Intelligence      │
└─────────────────────────────────────────┘
```

---

## 📚 Agent Modes

### Baseline Mode
- **Tools:** `search_documents` only
- **Behavior:** Plain RAG — search documents, answer from text
- **Best for:** Simple factual questions
- **Example:** "What is the company about?"

### GraphRAG Mode
- **Tools:** `search_documents` + `explore_entity` + `query_graph`
- **Behavior:** Search text AND knowledge graph
- **Best for:** Complex questions, comparisons, "list all…"
- **Example:** "Compare the premium and standard health plans"

---

## 🌍 Languages Supported

The system supports 4 languages:
- 🇬🇧 English (`en`)
- 🇩🇪 Deutsch (`de`)
- 🇱🇻 Latviešu (`lv`)
- 🇪🇸 Español (`es`)

Select in the UI sidebar.

---

## 💾 Data Storage

| Path | Purpose |
|---|---|
| `data/raw/` | Sample Zava documents (PDFs, markdown) |
| `data/cache/` | Cached processing results (embeddings, extractions) |
| `data/history.sqlite3` | Chat conversations and messages |

---

## 📖 Additional Resources

- **Full architecture:** [`docs/PLAN.md`](docs/PLAN.md)
- **Graph schema:** [`docs/ontology.md`](docs/ontology.md)
- **Code structure:** [`CLAUDE.md`](CLAUDE.md)
- **Dependencies analysis:** [`GRAPHRAG_DEPENDENCIES.md`](GRAPHRAG_DEPENDENCIES.md)

---

## ✅ Checklist

Before running:
- [ ] Python 3.12+ installed
- [ ] `uv` installed
- [ ] Docker running
- [ ] Git repo cloned
- [ ] `uv sync` completed
- [ ] `.env` configured with Foundry credentials
- [ ] `.env.database-mcp` configured with Neo4j credentials
- [ ] Neo4j container started
- [ ] Logged into Azure (`az login`)

When running:
- [ ] Terminal 1: Database MCP Server running
- [ ] Terminal 2: Chatbot Backend running
- [ ] Terminal 3: Streamlit UI running
- [ ] Browser open to http://localhost:8501

---

**Happy testing! 🚀**
