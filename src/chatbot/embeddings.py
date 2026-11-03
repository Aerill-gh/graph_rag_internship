"""Embedding generation for the chatbot, independent of graphrag."""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path


def _required(name: str) -> str:
    """Get a required environment variable."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name}. Add it to .env file.")
    return value


def _load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def get_azure_openai_api_key() -> str:
    """Get Azure OpenAI API key."""
    _load_env()
    return _required("AZURE_OPENAI_API_KEY")


def get_azure_openai_base_url() -> str:
    """Get Azure OpenAI base URL."""
    _load_env()
    return _required("AZURE_OPENAI_BASE_URL")


def get_embedding_deployment() -> str:
    """Get Azure embedding model deployment name."""
    _load_env()
    return _required("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding using Azure OpenAI.
    
    This is independent of graphrag and can be used by the chatbot directly.
    """
    if not text.strip():
        raise ValueError("Cannot generate an embedding for empty text.")

    api_key = get_azure_openai_api_key()
    base_url = get_azure_openai_base_url()
    deployment = get_embedding_deployment()

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    response = client.embeddings.create(
        model=deployment,
        input=text,
    )

    return response.data[0].embedding
