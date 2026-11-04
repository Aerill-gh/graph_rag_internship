# System Architecture & Communication Guide

Understanding how all components work together.

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER LAYER                               │
│  Streamlit Web UI (http://localhost:8501)                   │
│  • Chat interface                                           │
│  • Conversation management                                  │
│  • Language selection                                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │ REST API (HTTP/JSON)      │
        │ /chat, /conversations     │
        │ /languages, etc.          │
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                 APPLICATION LAYER                           │
│  Chatbot Backend (FastAPI) (http://localhost:8000)          │
│  • Agent orchestration (BaselineAgent, GraphRAGAgent)       │
│  • Chat history management                                  │
│  • Tool invocation coordination                             │
│  • Chat store (SQLite: data/history.sqlite3)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────────────┐
        │ MCP (JSON-RPC over HTTP)           │
        │ search_documents                   │
        │ explore_entity                     │
        │ query_graph                        │
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              DATABASE LAYER                                 │
│  Database MCP Server (http://localhost:8002)                │
│  • MCP protocol server                                      │
│  • Tool implementation wrappers                             │
│  • Neo4j connection pooling                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────────────┐
        │ Bolt Protocol (Neo4j native)       │
        │ Cypher queries                     │
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              STORAGE LAYER                                  │
│  Neo4j Database (bolt://localhost:7687)                     │
│  • Knowledge graph (entities, relationships)                │
│  • Vector index (embeddings)                                │
│  • Document metadata                                        │
└─────────────────────────────────────────────────────────────┘

EXTERNAL (Azure Cloud):
┌─────────────────────────────────────────────────────────────┐
│  Microsoft Foundry / Azure AI Services                      │
│  • Chat model (GPT-4 or similar) - for answers & extraction │
│  • Embedding model - for vector generation                  │
│  • Document Intelligence - for PDF parsing                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Component Interactions

### 1. User Asks a Question

**Flow:**
```
UI → POST /chat {message, mode, language}
        ↓
Chatbot Backend receives request
        ↓
Agent selected (BaselineAgent or GraphRAGAgent)
        ↓
Agent.ask() is called with the message
```

### 2. Agent Processes the Question

**BaselineAgent:**
```
Agent receives: message, language, history
        ↓
[1] Generate embedding for question
        ├─ Uses: graphrag.models.generate_embedding()
        ├─ Calls: Azure Foundry embedding model
        ├─ Returns: embedding vector
        │
[2] Run search_documents tool
        ├─ Calls: DatabaseMcpClient.search_documents(embedding, top_k=5)
        ├─ Calls: MCP Server (localhost:8002)
        ├─ MCP Server calls: Neo4j query
        ├─ Returns: [Document, page, chunk, similarity]
        │
[3] Generate answer
        ├─ Calls: Azure Foundry chat model
        ├─ Prompt: Include search results + question
        ├─ Model generates: Natural language answer
        │
[4] Format response
        ├─ Extract citations (which chunks were used)
        ├─ Return: ChatResponse {answer, sources, tools_used}
        │
Response → HTTP 200 with answer + sources
```

**GraphRAGAgent:**
```
Agent receives: message, language, history
        ↓
[1] Search documents (same as Baseline)
        ├─ Generate embedding
        ├─ Call search_documents via MCP
        ├─ Get text passages
        │
[2] Run graph tools (based on agent decision)
        ├─ explore_entity(name, label)
        │  └─ Find connected entities in graph
        ├─ query_graph(pattern, sources, target)
        │  └─ Find relationships between entities
        ├─ Both call: DatabaseMcpClient via MCP
        ├─ Both call: Neo4j for queries
        │
[3] Combine results
        ├─ Merge: text passages + graph facts
        ├─ De-duplicate results
        │
[4] Generate answer (with more context than Baseline)
        ├─ Better for: comparisons, lists, reasoning chains
        │
Response → HTTP 200 with answer + sources
```

### 3. Tool Execution (via MCP)

**When chatbot calls a tool:**

```
Agent: tool_name = "search_documents", arguments = {embedding, top_k}
        ↓
BaselineAgent.run_tool("search_documents", {embedding, top_k})
        ↓
Create MCP client:
  client = DatabaseMcpClient()
        ↓
Call MCP method:
  result = client.search_documents(embedding, top_k)
        ↓
MCP Client makes HTTP POST to MCP Server (localhost:8002)
  POST /mcp with JSON-RPC request:
  {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search_documents",
      "arguments": {embedding: [...], top_k: 5}
    }
  }
        ↓
MCP Server receives, executes tool:
  search_documents(embedding, top_k)
    ├─ Connect to Neo4j (bolt://localhost:7687)
    ├─ Run similarity search query:
    │  MATCH (chunk:Chunk) 
    │  WHERE vector.similarity(chunk.embedding, $vec) > 0.7
    │  RETURN chunk, similarity ORDER BY similarity DESC
    │  LIMIT $limit
    ├─ Format results with sources
    └─ Return to client
        ↓
MCP Server returns JSON-RPC response:
  {
    "jsonrpc": "2.0",
    "result": {
      "passages": [
        {
          "text": "...",
          "source_document": "...",
          "page": 5,
          "chunk_id": "chunk_123"
        }
      ]
    }
  }
        ↓
MCP Client parses response:
  SearchDocumentsResponse(passages=[...])
        ↓
Agent receives tool results
```

### 4. Save Conversation

**When user's question is answered:**

```
Agent returns answer
        ↓
Chatbot backend receives response
        ↓
If conversation_id provided in request:
        ├─ Store user message: store.add_message(conv_id, "user", message)
        └─ Store answer: store.add_message(conv_id, "assistant", answer)
        ↓
ChatStore writes to SQLite (data/history.sqlite3)
        ├─ Table: conversations (id, title, created_at, pinned, language)
        └─ Table: messages (id, conversation_id, role, content, timestamp)
        ↓
UI fetches updated conversation list
  GET /conversations → Returns list of all chats
        ↓
UI displays in sidebar with timestamps and previews
```

---

## 📡 Communication Protocols

### 1. UI ↔ Chatbot Backend
**Protocol:** HTTP REST (JSON)
**Port:** 8000
**Endpoints:**
- `POST /chat` - Send message, get answer
- `GET /conversations` - List all chats
- `POST /conversations` - Create new chat
- `GET /conversations/{id}/messages` - Get chat history
- `PATCH /conversations/{id}` - Update (rename, pin, language)
- `DELETE /conversations/{id}` - Delete chat
- `GET /languages` - Get available languages

### 2. Chatbot Backend ↔ Database MCP Server
**Protocol:** JSON-RPC 2.0 over HTTP
**Port:** 8002
**Methods (Tools):**
- `search_documents` - Vector similarity search in Neo4j
- `explore_entity` - Find connected entities in graph
- `query_graph` - Find relationships (Text2Cypher)

### 3. Database MCP Server ↔ Neo4j
**Protocol:** Bolt (Neo4j's binary protocol)
**Port:** 7687
**Operations:**
- Read vector similarity: `vector.similarity(a, b) > threshold`
- Cypher queries: `MATCH`, `WHERE`, `RETURN`
- No write operations (read-only from MCP)

---

## 🔐 Data Flow Examples

### Example 1: Simple Question (Baseline Mode)

```
User types: "What is the company about?"
    ↓
UI: POST /chat {message: "...", mode: "baseline", language: "en"}
    ↓
Chatbot Backend → BaselineAgent.ask(message)
    ↓
Generate embedding for "What is the company about?"
    ├─ Calls: graphrag.models.generate_embedding()
    ├─ Azure Foundry embedding model
    └─ Returns: [0.123, -0.456, 0.789, ...]
    ↓
Call search_documents(embedding, top_k=5)
    ├─ DatabaseMcpClient().search_documents(embedding, 5)
    ├─ HTTP POST to MCP Server (localhost:8002)
    ├─ MCP Server queries Neo4j (localhost:7687)
    ├─ Neo4j returns top 5 chunks by similarity
    └─ Returns: [{text, source, page, chunk_id}, ...]
    ↓
Call chat model with results
    ├─ Azure Foundry chat model
    ├─ Prompt: "Answer this question using these documents: ..."
    ├─ Model generates: "Zava is a fictional company focused on..."
    └─ Returns: answer text
    ↓
Extract citations from chunks used
    ├─ Each result has: source_document, page, chunk_id
    └─ Format: [{citation_id, document, pages, chunk_id}, ...]
    ↓
Return ChatResponse {answer, sources, tools_used: ["search_documents"]}
    ↓
UI receives and displays answer with sources
```

### Example 2: Complex Question (GraphRAG Mode)

```
User types: "Compare the premium and standard health plans"
    ↓
UI: POST /chat {message: "...", mode: "graphrag", language: "en"}
    ↓
Chatbot Backend → GraphRAGAgent.ask(message)
    ↓
Generate embedding, call search_documents (same as Baseline)
    ├─ Get passages about health plans
    └─ Returns: text passages
    ↓
Agent analyzes and calls explore_entity("Northwind Health Plus", "Plan")
    ├─ DatabaseMcpClient().explore_entity("Northwind Health Plus", "Plan", limit=10)
    ├─ MCP Server queries Neo4j:
    │  MATCH (e:Plan {name: "Northwind Health Plus"})
    │  MATCH (e)-[*1..2]-(related)
    │  RETURN e, related, type(relationships)
    ├─ Returns: Entity and connected nodes
    └─ Results: Features, benefits, coverage details
    ↓
Agent calls explore_entity("Northwind Standard Benefits", "Plan")
    ├─ Same process for standard plan
    └─ Results: Features, benefits, coverage details
    ↓
Combine all results (text + graph facts)
    ├─ Premium plan facts from graph
    ├─ Standard plan facts from graph
    ├─ Original search passages
    ├─ De-duplicate
    └─ Organize by comparison points
    ↓
Call chat model with enriched context
    ├─ Prompt: "Compare these two plans: ..." (with all facts)
    ├─ Model generates: Detailed comparison
    └─ Returns: answer text
    ↓
Extract citations (from both text and graph)
    ├─ Text sources: documents, pages, chunks
    ├─ Graph sources: entities found in Neo4j
    └─ Combine: [{citation_id, document, pages, chunk_id}, ...]
    ↓
Return ChatResponse {answer, sources, tools_used: ["search_documents", "explore_entity"]}
    ↓
UI shows: "Used tools: search_documents → explore_entity"
```

---

## 🗂️ File Organization

```
src/
├── UI/
│   ├── streamlit_chat.py          ← Streamlit UI, calls Chatbot API
│   └── assets/                    ← Logos, styles
│
├── ui/
│   ├── locales.py                 ← UI text for all languages
│   └── ...
│
├── chatbot/
│   ├── chat_bot_main.py           ← FastAPI server, REST endpoints
│   ├── agent.py                   ← BaselineAgent, GraphRAGAgent
│   ├── chat_store.py              ← SQLite conversation storage
│   ├── config.py                  ← Foundry settings
│   ├── languages.py               ← Language definitions
│   ├── ask.py                     ← CLI tool (optional)
│   └── mcp_server.py              ← Deprecated (replaced by database_mcp)
│
└── graphrag/
    ├── models.py                  ← generate_embedding() function
    ├── tools.py                   ← Tool implementations (NOT imported by chatbot)
    ├── database_mcp/
    │   ├── server.py              ← MCP server (runs on :8002)
    │   ├── client.py              ← MCP client (used by chatbot)
    │   ├── contracts.py           ← Type definitions
    │   ├── config.py              ← Neo4j settings
    │   └── neo4j_backend.py       ← Neo4j integration
    │
    ├── config.py                  ← Project config
    ├── ontology.py                ← Graph schema
    ├── pipeline.py                ← Processing pipeline
    ├── extract*.py                ← Entity extraction
    ├── embeddings.py              ← Embedding functions
    ├── Neo4j/                     ← Neo4j storage
    ├── documents.py               ← Document processing
    ├── build_graph.py             ← Build graph script
    └── ...

data/
├── raw/                           ← Original documents (PDFs, markdown)
├── cache/                         ← Cached processing results
└── history.sqlite3               ← Chat conversation storage
```

---

## 🔍 How to Debug

### 1. Check if Services are Running

```bash
# Check ports in use
netstat -tuln | grep -E '8000|8002|7687'

# Or using curl
curl http://localhost:8000/languages        # Chatbot
curl http://localhost:8002/mcp              # MCP Server
curl http://localhost:7474                  # Neo4j Browser
```

### 2. View Logs

```bash
# In Terminal 1 (MCP Server)
# Logs show: MCP tool calls, Neo4j queries

# In Terminal 2 (Chatbot Backend)
# Logs show: HTTP requests, agent reasoning, tool calls

# In Terminal 3 (Streamlit UI)
# Logs show: User actions, API calls
```

### 3. Inspect Database

```bash
# Open Neo4j Browser
http://localhost:7474

# Example queries:
MATCH (n) RETURN count(n) as node_count
MATCH (n:Chunk) RETURN n LIMIT 5
MATCH (n:Entity) RETURN n LIMIT 10
```

### 4. Test MCP Tools Directly

```bash
# Test search_documents
curl -X POST http://localhost:8002/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search_documents",
      "arguments": {
        "embedding": [0.1, 0.2, 0.3, ...],
        "top_k": 5
      }
    }
  }'
```

### 5. Test Chatbot Endpoints

```bash
# List conversations
curl http://localhost:8000/conversations

# Get languages
curl http://localhost:8000/languages

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the company about?",
    "mode": "baseline",
    "language": "en"
  }'
```

---

## 🚨 Common Issues & Solutions

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| "Cannot reach chatbot" | Backend not running | Start Terminal 2 |
| "NEO4J_URI not set" | `.env.database-mcp` missing | Copy `.env.database-mcp.example`, fill in values |
| "Neo4j connection refused" | Neo4j not running | `docker start neo4j-zava` |
| "MCP timeout" | MCP Server not running | Start Terminal 1 |
| "Embedding model not found" | Foundry config wrong | Check FOUNDRY_EMBEDDING_DEPLOYMENT in `.env` |
| "No results found" | Knowledge graph empty | Run `build_graph.py` to populate Neo4j |
| "ClosedResourceError in MCP" | MCP initialization failed | Check Neo4j credentials in `.env.database-mcp` |

---

## ✅ Health Check Script

```bash
#!/bin/bash
echo "Checking services..."

echo -n "Neo4j... "
curl -s http://localhost:7474 > /dev/null && echo "✓" || echo "✗"

echo -n "Database MCP Server... "
curl -s http://localhost:8002/mcp > /dev/null && echo "✓" || echo "✗"

echo -n "Chatbot Backend... "
curl -s http://localhost:8000/languages > /dev/null && echo "✓" || echo "✗"

echo -n "Streamlit UI... "
curl -s http://localhost:8501 > /dev/null && echo "✓" || echo "✗"

echo "Done!"
```

Save as `health_check.sh`, run with `bash health_check.sh`.

---

**For more details, see:**
- [`QUICKSTART.md`](QUICKSTART.md) - Detailed setup
- [`QUICKRUN.md`](QUICKRUN.md) - Fast run for experienced devs
- [`docs/PLAN.md`](docs/PLAN.md) - Full project plan
- [`docs/ontology.md`](docs/ontology.md) - Graph schema
