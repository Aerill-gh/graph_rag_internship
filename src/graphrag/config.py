import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_DIR = PROJECT_ROOT / "document_storage"
load_dotenv(PROJECT_ROOT / ".env")


def get_rest_api_port() -> int:
    try:
        port = int(os.getenv("GRAPHRAG_PORT", "").strip() or "8001")
    except ValueError:
        raise ValueError("GRAPHRAG_PORT must be an integer port.") from None
    if not 1 <= port <= 65535:
        raise ValueError("GRAPHRAG_PORT must be between 1 and 65535.")
    return port


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Add it to the project .env file before running this script."
        )
    return value


def _required_deployment(name: str) -> str:
    value = _required_env(name)
    if "://" in value or "/" in value:
        raise RuntimeError(
            f"{name} must be a model deployment name, not an endpoint URL. "
            "Copy the deployed model's Name from Foundry."
        )
    return value


@dataclass(frozen=True)
class AzureOpenAISettings:
    api_key: str = field(repr=False)
    base_url: str
    deployment: str


@dataclass(frozen=True)
class FoundrySettings:
    project_endpoint: str
    chat_deployment: str


@dataclass(frozen=True)
class Settings:
    docintel_endpoint: str = field(
        default_factory=lambda: os.getenv("DOCINTEL_ENDPOINT")
        or "https://graphrag.cognitiveservices.azure.com/"
    )
    @property
    def foundry(self) -> FoundrySettings:
        return FoundrySettings(
            project_endpoint=_required_env("FOUNDRY_PROJECT_ENDPOINT"),
            chat_deployment=_required_deployment("FOUNDRY_CHAT_DEPLOYMENT"),
        )

    @property
    def azure_openai(self) -> AzureOpenAISettings:
        return AzureOpenAISettings(
            api_key=_required_env("AZURE_OPENAI_API_KEY"),
            base_url=_required_env("AZURE_OPENAI_BASE_URL"),
            deployment=_required_env("AZURE_OPENAI_DEPLOYMENT"),
        )

    @property
    def azure_openai_embedding_deployment(self) -> str:
        return _required_deployment("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


settings = Settings()