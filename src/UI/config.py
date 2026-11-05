
import os
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _required(name: str) -> str:
    """Get a required environment variable, raise error if missing or empty."""
    _load_env()
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required configuration: {name}. "
            f"Add it to the .env file."
        )
    return value


def _port(name: str) -> int:
    """Get a required port configuration from environment."""
    value = _required(name)
    try:
        port = int(value)
    except ValueError:
        raise ValueError(f"{name} must be a valid integer, got: {value}") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535, got: {port}")
    return port


def get_ui_port() -> int:
    """Get the Streamlit UI port (required). No default provided."""
    return _port("UI_PORT")


def get_chatbot_api_url() -> str:
    """Get the chatbot API endpoint URL (required). No default provided."""
    return _required("CHATBOT_API_URL")


def get_backend_api_url() -> str:
    """Get the backend API endpoint URL (required). No default provided."""
    return _required("BACKEND_API_URL")
