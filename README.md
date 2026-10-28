# Zava GraphRAG Agent — Internship Project

An internship project to build a **GraphRAG AI agent** on Azure, using **Microsoft Foundry**, **Foundry Agent Service**, **Azure AI Document Intelligence** and **Neo4j**.

**GraphRAG** adds a knowledge graph to normal RAG. The LLM turns documents into entities and relationships, which are stored in a graph. Questions are then answered using both similar text and connected facts from the graph. This helps with questions that plain RAG struggles with, such as comparisons and "list all…" questions.

> 📋 **Full plan, architecture and weekly tasks:** [`docs/PLAN.md`](docs/PLAN.md)
>
> **Status:** 🟡 Kickoff. The folders and commands below are planned and don't exist yet.

---

## Goal

Build an **AI agent** for **Zava**, a fictional company, that answers employee questions about **roles, policies and benefits**. It answers only from Zava's documents and shows its sources. The agent uses tools to search a Neo4j knowledge graph and chooses the right tool for each question. Then show that GraphRAG answers better than plain RAG.

---

## Architecture

![Zava GraphRAG Agent architecture](docs/architecture.svg)

| Part | Service / tool |
|---|---|
| AI platform | **Microsoft Foundry** project (formerly Azure AI Foundry) |
| Models | Chat model + embedding model, deployed in Foundry |
| Agent | **Foundry Agent Service**, with tools: `search_documents`, `explore_entity`, `query_graph` |
| PDF reading | **Azure AI Document Intelligence** (layout model) |
| Graph + vector database | **Neo4j** in Docker, locally |
| Code | **Python 3.12** with **uv**, `azure-ai-projects` SDK, `neo4j` driver |
| Chat UI | **Streamlit** |

The app and Neo4j run locally. The AI services run in Azure. Details are in [`docs/PLAN.md`](docs/PLAN.md#2-architecture).

---

## Data

We use the sample documents from Microsoft's official Azure RAG sample, [`Azure-Samples/azure-search-openai-demo`](https://github.com/Azure-Samples/azure-search-openai-demo/tree/main/data). They describe **Zava**, a fictional company. MIT license, 7 files, about 4 MB.

| File | Content |
|---|---|
| `Zava_Company_Overview.md` | What the company is and does |
| `employee_handbook.pdf` | Company policies, rules, processes |
| `role_library.pdf` | Job roles, responsibilities, skills |
| `Benefit_Options.pdf` | Overview of employee benefits |
| `Northwind_Health_Plus_Benefits_Details.pdf` | Premium health plan details |
| `Northwind_Standard_Benefits_Details.pdf` | Standard health plan details |
| `PerksPlus.pdf` | Employee perks programme |

### Graph model (starting point)

```
(Document)-[:HAS_CHUNK]->(Chunk)          // text + embedding
(Chunk)-[:MENTIONS]->(entity)             // links facts to their source

(Role)-[:BELONGS_TO]->(Department)
(Role)-[:REQUIRES]->(Skill)
(Policy)-[:APPLIES_TO]->(Role)
(Plan)-[:COVERS]->(Service)
```

### Example questions

| Question | Why the graph helps |
|---|---|
| Which services does Health Plus cover that Standard doesn't? | Compares two documents |
| What skills do engineering roles have in common? | Collects facts from many roles |
| Which policies apply to managers? | Follows relationships |

---

## Project structure (planned)

```
internship/
├── src/graphrag/       # config, documents, models, extract, graph, tools, agent
├── scripts/            # download_data.py, build_graph.py, ask.py
├── app.py              # Streamlit chat UI
├── evaluation/         # Test questions + comparison script
├── data/               # raw/ documents + cache/ of saved results (not in git)
└── docs/               # PLAN.md, graph schema, weekly notes
```

---

## Getting started

### You need
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- [Docker](https://www.docker.com/) (for Neo4j)
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), signed in with `az login`
- Access to the team's **Foundry project** and **Document Intelligence** resource (from the lead)

### Setup (planned)

Install dependencies:
```bash
uv sync
```

Create your local config and fill in the values:
```bash
cp .env.example .env
```

Start Neo4j:
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/<password> neo4j:5
```

Download the documents:
```bash
uv run python scripts/download_data.py
```

Build the graph:
```bash
uv run python scripts/build_graph.py
```

Start the chat UI:
```bash
uv run streamlit run app.py
```

### Local chatbot API prototype

The API supports two backends selected with `CHATBOT_BACKEND` in `.env`:

- `foundry` (default): uses Foundry Agent Service with the `search_documents`
  tool. It reuses a matching agent version or creates an updated version when the
  model, instructions, or tools change. Each request references that exact version;
  the app does not change a shared agent endpoint's traffic routing.
- `langchain`: runs a LangChain agent locally, calling the chat deployment through
  the Foundry project endpoint with Azure sign-in. It does not create a hosted agent.

Both require `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL_NAME` (the chat deployment
name). `FOUNDRY_AGENT_NAME` is used only by the Foundry backend.

From the repository root:

```powershell
uv sync
az login
# Start the MCP server in another terminal first (instructions below).
uv run python -m Chatbot.chat_bot_main
```

Open `http://127.0.0.1:8000/docs` and try `POST /chat` with
`{"message": "Hello"}`. The response is `{"answer": "..."}`.

### Streamlit chatbot UI

Run the FastAPI chatbot first using the instructions above. For RAG mode, keep
Neo4j and the MCP server running as well. From the repository root, start the UI:

```powershell
uv sync
uv run streamlit run UI\streamlit_chat.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).
Use the top-right menu to select **Light**, **Dark**, or **System**. The chat
background, text, borders, and accents follow the selected theme.
The UI sends `{"message": "...", "history": [...]}` to `CHATBOT_API_URL`, defaulting to
`http://127.0.0.1:8000/chat`. Set `CHATBOT_API_URL` in `.env` only if the API
uses a different address. Restart both the API and UI after updating to this version.

- **New chat** creates a separate conversation. Select a chat in the sidebar to switch.
- Titles start as **New chat** and automatically use the first question (up to 60
  characters), without an extra AI call.
- Open the **three-dot menu** next to a chat for **Pin/Unpin**, **Rename**, or
  **Delete**. Rename opens a separate name dialog; Delete opens a confirmation
  dialog and removes that chat's locally saved messages only after confirmation.
- At most **three** chats can be pinned. Pinned chats appear first; within each
  group, recently active chats appear first.

Chats, names, pins, and messages are saved in `data/chats/history.sqlite3`, which
is git-ignored. They survive browser refreshes and app restarts. This is a local,
single-user library: all browsers connecting to this app share it. There are no
accounts or per-user access controls. Messages are stored as plain text on this
computer, so do not use it for sensitive employee information.

Each UI request includes only the selected chat's saved user/assistant messages.
The API uses an isolated context for that request, rather than its default
conversation, so chats do not share AI memory and follow-ups still work after an
API restart. Foundry creates a fresh hosted conversation for each such request;
deleting a local chat does not delete hosted Foundry conversations. Long chats
send more context and consume more tokens. Tool results are still retrieved anew
for every RAG turn. Requests without `history` retain the original default API
conversation behavior; use `"history": []` to explicitly start independently.

To run Foundry **without MCP or document search**, use:

```powershell
$env:CHATBOT_BACKEND = "foundry"
$env:CHATBOT_USE_DOCUMENTS = "false"
uv run python -m Chatbot.chat_bot_main
```

Alternatively set `CHATBOT_USE_DOCUMENTS=false` in `.env`. Chat-only mode registers
no search tools, provides no document citations, and answers from the model's
general knowledge. It still avoids inventing Zava-specific facts. Foundry settings
and Azure sign-in are still required. Set `CHATBOT_USE_DOCUMENTS=true` to restore
document-grounded mode (the default); this requires the MCP server.

This is a local learning prototype, not a multi-user chat service. The default
API conversation (requests without `history`) is shared; its LangChain history
is kept in memory and lost when the app restarts.
In document-grounded mode, the Foundry backend requests document context from the MCP service before
answering each message. It does not import retrieval code or access Neo4j. The LangChain
backend is still chat-only in this entry point. `LangChainChatBot`
accepts optional LangChain tools and a checkpointer; its caller owns and closes
the model clients (the API does this during shutdown). Tool execution is bounded
by a graph recursion limit. Do not enable external LangSmith tracing for project
data.

Run the offline chatbot tests with:

```powershell
uv run python -m unittest discover -s tests -p test_chat_bot.py
uv run python -m unittest discover -s tests -p test_foundry_tools.py
uv run python -m unittest discover -s tests -p test_mcp_server.py
uv run python -m unittest discover -s tests -p test_streamlit_chat.py
uv run python -m unittest discover -s tests -p test_saved_chats.py
```

### Local MCP document-search server

The chatbot owns the REST endpoint, agent instructions, tool-call loop, MCP client,
MCP server, answer citations, and the `graphrag.tools.search_documents` tool.
The RAG team supplies document parsing, embedding helpers, ingestion, and graph
storage. The MCP server delegates to the search tool.

The flow is: question -> Foundry function call -> MCP document-search tool ->
passages returned to Foundry -> answer with `[S1]` citations and a source list.

Start the MCP server from the repository root in a separate PowerShell terminal:

```powershell
# The repository's src/ tree is not installed as a Python package yet.
$env:PYTHONPATH = (Resolve-Path .\src).Path
uv run python -m Chatbot.mcp_server
```

The server binds to `127.0.0.1:8001` and exposes Streamable HTTP at `/mcp`.
It is local-only and has no authentication; do not expose it to the network.
Set these values in `.env`, then start the chatbot in another terminal:

```dotenv
CHATBOT_BACKEND=foundry
MCP_SERVER_URL=http://127.0.0.1:8001/mcp
MCP_SEARCH_TOOL=search_documents
```

The tool accepts `{"question": "..."}` and returns structured content:

```json
{"passages": [{
  "chunk_id": "example-chunk",
  "document": "example.pdf",
  "pages": [1],
  "text": "A source passage."
}]}
```

The server retains the source fields and omits the retrieval similarity score.
An empty passages list means no evidence; Markdown may have an empty page list.
Blank questions, retrieval failures, and malformed results are MCP errors, not
empty results. Synchronous retrieval runs in a worker thread.
The client also accepts one text block containing a JSON array of passages.

Set `CHATBOT_BACKEND=foundry`, `MCP_SERVER_URL`, `FOUNDRY_PROJECT_ENDPOINT`, and
`FOUNDRY_MODEL_NAME` to use this path. Startup fails explicitly without the MCP URL.
The MCP server must be running before the chatbot starts.

**Retrieval prerequisites:** `graphrag.models` and `graphrag.neo4j_store`
import `graphrag.config.settings`. This configuration module is currently
git-ignored: obtain the team's configuration module for a fresh checkout and set
its required variables in your local `.env`. Retrieval requires embedding
configuration, Neo4j credentials, indexed chunks, and an online
`chunk_embedding_index` vector index. MCP startup and tool discovery work without
retrieval imports; a real search reports an error if its prerequisites are not
ready. The MCP wrapper does not implement or replace them.

Offline tests mock retrieval and cover tool discovery, validation, errors, source
preservation, and a real local Streamable HTTP round trip. They make no Azure or
Neo4j calls. For a live check once retrieval is ready, ask a question with an answer
you can verify in a source document, then check the passage text, page, and citation.
The running app never returns fake documents.

The Foundry loop forces a search before answering and permits at most three tool
calls per message. MCP failures and malformed responses are errors, not empty
search results. Final answers must cite known source IDs, or explicitly decline
for insufficient evidence. Source-ID validation does not prove each claim is
supported: check answers against the actual passages.

A failed turn starts a fresh conversation on the next request. This prototype
does not automatically delete persisted Foundry conversations.

### `.env`
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=

# Microsoft Foundry
FOUNDRY_PROJECT_ENDPOINT=
FOUNDRY_CHAT_DEPLOYMENT=
FOUNDRY_EMBEDDING_DEPLOYMENT=

# Document Intelligence
DOCINTEL_ENDPOINT=
DOCINTEL_KEY=
```

> ⚠️ Never commit `.env`. It's in `.gitignore`.

---

## Roadmap (5 weeks)

| Weeks | Focus | Done when |
|---|---|---|
| 1 | Setup: Foundry project, models, Document Intelligence, Neo4j | Everyone can run Neo4j and call a Foundry model |
| 2 | Learn Neo4j + Foundry, build a plain RAG agent | The agent answers with sources |
| 3 | Test questions + graph design | 20–30 test questions and the graph model agreed |
| 3 | Build the knowledge graph | One command builds the graph |
| 4 | GraphRAG agent: graph tools + comparison | GraphRAG beats plain RAG on comparison and list questions |
| 5 | Chat UI + final demo | Live demo works |

Details for each week are in [`docs/PLAN.md`](docs/PLAN.md#3-work-plan).

---

## Team workflow

- Work in `feature/<short-name>` branches and merge through a Pull Request with one review.
- Keep PRs small.
- Notebooks are fine for experiments. Move code you reuse into `src/`.
- Keep Azure costs down: save Document Intelligence and LLM results to `data/cache/` so you don't pay for them twice.

---

## Learning resources

- [Neo4j GraphAcademy](https://graphacademy.neo4j.com/): *Neo4j Fundamentals*, *Cypher Fundamentals*, *Building Knowledge Graphs with LLMs*
- [Microsoft Foundry docs](https://learn.microsoft.com/azure/foundry/)
- [Foundry Agent Service](https://learn.microsoft.com/azure/foundry/agents/overview) and [function calling](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/function-calling)
- [Azure AI Document Intelligence](https://learn.microsoft.com/azure/ai-services/document-intelligence/)
- [Neo4j vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Streamlit docs](https://docs.streamlit.io/)

---

## Team

- **Internship Lead:** _name / contact_
- **Interns:** _to be added_
