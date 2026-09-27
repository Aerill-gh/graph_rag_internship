# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project context

- An **internship learning project**: a **GraphRAG AI agent** using **Microsoft Foundry**, **Foundry Agent Service**, **Azure AI Document Intelligence** and **Neo4j**. See `README.md` for the overview and `docs/PLAN.md` for the architecture and weekly tasks.
- **This is not a production system.** Keep solutions simple. The Azure AI services above are part of the project. Don't add production operations layers (hosting setup, infrastructure as code, CI/CD, Key Vault, monitoring, API frameworks) unless the lead asks.
- The people writing code here are **interns who are learning**. The internship lead reviews their work.
- The project is at an early stage. Check what actually exists before assuming a file or command is there.

## Use case and data

- **Use case:** an AI agent for **Zava**, a fictional company. It answers employee questions about roles, policies and benefits, and shows its sources.
- **Data:** the 7 sample documents from Microsoft's [`Azure-Samples/azure-search-openai-demo`](https://github.com/Azure-Samples/azure-search-openai-demo/tree/main/data) `data/` folder (MIT license), stored in `data/raw/` (git-ignored). Don't add other data sources unless the lead asks.
- **Graph model:** the starting point is in `README.md` (`Document`, `Chunk`, `Role`, `Department`, `Skill`, `Policy`, `Plan`, `Service`). Keep extraction limited to it. If a new label or relationship seems needed, suggest it rather than adding it silently, and update `docs/graph-schema.md` once it's agreed.
- The documents are small, so any answer can be checked by reading the source. When debugging, point to the exact document and passage.

## How to work with interns

- **Explain, don't just do.** When you write or change code, briefly say *why* and which GraphRAG or Neo4j concept is behind it.
- Prefer small, readable changes. Plain, obvious code beats clever or abstract code.
- If asked to finish a whole task end to end, it's fine to help. Still leave the intern able to explain the result: add comments where the logic isn't obvious, and summarize what was done.

## Tech stack

- **Python 3.12+** with **uv** (`pyproject.toml`). Don't use pip, conda or `requirements.txt`.
- **Microsoft Foundry** project: chat and embedding models are deployed there. Call them through the Foundry project, not the public OpenAI endpoint.
- **Foundry Agent Service** for the agent, via the `azure-ai-projects` SDK. The agent's tools are Python functions in `src/graphrag/tools.py` (function calling). Check the current Foundry docs before using SDK APIs, because they change often.
- **Azure AI Document Intelligence** (layout model) turns PDFs into Markdown.
- **Neo4j 5.x** in Docker, with the official `neo4j` Python driver and a native vector index.
- **Streamlit** for the chat UI.
- Build the GraphRAG parts (chunking, extraction, graph queries) in **plain Python**, not a GraphRAG framework, so each step is visible. If `pyproject.toml` already contains a framework, follow it.

## Commands

These are planned. Update this section once they exist.

```bash
uv sync                                   # install dependencies
uv run python scripts/download_data.py    # fetch the 7 documents into data/raw/
uv run python scripts/build_graph.py      # chunk, embed, extract, save to Neo4j
uv run python scripts/ask.py "question"   # ask the agent from the terminal
uv run streamlit run app.py               # chat UI
uv run python evaluation/run_eval.py      # plain RAG vs GraphRAG on the test questions
```

## Code conventions

- Code lives in `src/graphrag/` as a few simple modules (`config`, `documents`, `models`, `extract`, `graph`, `tools`, `agent`). Scripts in `scripts/` call them.
- Settings come from `.env` through `config.py`. Don't scatter `os.getenv` calls through the code.
- Keep LLM prompts, agent instructions and tool descriptions as clearly named constants or files, so they're easy to find and change. Tool descriptions matter: the agent picks tools based on them.
- The **plain RAG baseline** is the same agent with only the `search_documents` tool. Keep it working so the comparison stays fair.
- Tools return results with their source (document, page, chunk), and the agent's answers must cite them.
- **Save expensive results** (Document Intelligence output, embeddings, extraction output) to `data/cache/`, so re-running doesn't cost again.

## Neo4j rules

- **Always use query parameters** (`$param`). Never build Cypher by joining strings with user input.
- Use `MERGE` on a unique key when saving entities, to avoid duplicates.
- The `query_graph` tool (Text2Cypher) runs generated queries **read-only** (`execute_read`) with a `LIMIT`. Generated Cypher must never write to the graph.

## Secrets and cost

- **Never commit secrets.** `.env` is git-ignored. `.env.example` has the variable names with empty values.
- Never print or log keys or passwords, and don't read `.env` into the conversation unless explicitly asked.
- Don't create or change Azure resources (Foundry projects, model deployments, Document Intelligence) without confirmation from the user. Creating and deleting agents in the team's Foundry project from code is fine.
- Keep costs low: use small test runs first, and avoid unbounded loops of LLM calls.

## Testing

- The main check is the **test question set** in `evaluation/`. After changing prompts, chunking or retrieval, re-run it and report the before and after results.
- Unit tests with `pytest` are welcome for simple logic (chunking, duplicate merging). Mock the LLM so tests cost nothing.

## Git

- Work on `feature/<short-name>` branches. Don't commit directly to `main`.
- Only commit or push when the user asks.
