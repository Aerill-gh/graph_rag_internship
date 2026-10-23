import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class ChatSettings:
    backend: str
    endpoint: str
    model_name: str
    agent_name: str
    mcp_server_url: str = ""
    mcp_search_tool: str = "search_documents"
    use_documents: bool = True


def load_chat_settings() -> ChatSettings:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    backend = os.getenv("CHATBOT_BACKEND", "foundry").strip().lower()
    if backend not in ("foundry", "langchain"):
        raise ValueError("CHATBOT_BACKEND must be 'foundry' or 'langchain'.")

    endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    model_name = os.getenv("FOUNDRY_MODEL_NAME", "").strip()
    if not endpoint or not model_name:
        raise RuntimeError(
            "Set FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL_NAME in your .env file."
        )
    mcp_server_url = os.getenv("MCP_SERVER_URL", "").strip()
    use_documents_value = os.getenv("CHATBOT_USE_DOCUMENTS", "true").strip().lower()
    if use_documents_value not in ("true", "false"):
        raise ValueError("CHATBOT_USE_DOCUMENTS must be 'true' or 'false'.")
    use_documents = use_documents_value == "true"
    if backend == "foundry" and use_documents and not mcp_server_url:
        raise RuntimeError(
            "Set MCP_SERVER_URL to the local Streamable HTTP MCP endpoint "
            "(http://127.0.0.1:8001/mcp). "
            "Document retrieval cannot run without it."
        )

    return ChatSettings(
        backend=backend,
        endpoint=endpoint,
        model_name=model_name,
        agent_name=os.getenv("FOUNDRY_AGENT_NAME", "zava-chatbot"),
        mcp_server_url=mcp_server_url,
        mcp_search_tool=os.getenv("MCP_SEARCH_TOOL", "search_documents"),
        use_documents=use_documents,
    )
