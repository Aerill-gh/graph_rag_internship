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


@dataclass(frozen=True)
class FoundrySettings:
    """Foundry project configuration for the chatbot agent."""
    project_endpoint: str
    chat_deployment: str


def _load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _port(name: str, default: int) -> int:
    _load_env()
    try:
        port = int(os.getenv(name, "").strip() or str(default))
    except ValueError:
        raise ValueError(f"{name} must be an integer port.") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535.")
    return port


def get_rest_api_port() -> int:
    return _port("CHATBOT_PORT", 8000)


def get_ui_port() -> int:
    return _port("UI_PORT", 8501)


def get_api_urls() -> tuple[str, str]:
    base_url = f"http://127.0.0.1:{get_rest_api_port()}"
    return (
        os.getenv("CHATBOT_API_URL", "").strip() or f"{base_url}/chat",
        os.getenv("BACKEND_API_URL", "").strip() or base_url,
    )


def _required(name: str) -> str:
    """Get a required environment variable, raise error if missing."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing {name}. Add it to the .env file."
        )
    return value


def get_foundry_endpoint() -> str:
    """Get Foundry project endpoint."""
    _load_env()
    return _required("FOUNDRY_PROJECT_ENDPOINT")


def get_foundry_chat_deployment() -> str:
    """Get Foundry chat model deployment name."""
    _load_env()
    return _required("FOUNDRY_CHAT_DEPLOYMENT")


def load_foundry_settings() -> FoundrySettings:
    """Load Foundry settings for the chatbot agent."""
    return FoundrySettings(
        project_endpoint=get_foundry_endpoint(),
        chat_deployment=get_foundry_chat_deployment(),
    )


def load_chat_settings() -> ChatSettings:
    """Load chat backend settings."""
    _load_env()
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
