import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


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
class Neo4jSettings:
    uri: str
    username: str
    password: str = field(repr=False)
    database: str = "neo4j"


@dataclass(frozen=True)
class Settings:
    docintel_endpoint: str = field(
        default_factory=lambda: os.getenv("DOCINTEL_ENDPOINT")
        or "https://graphrag.cognitiveservices.azure.com/"
    )
    storage_account_name: str = field(
        default_factory=lambda: os.getenv("STORAGE_ACCOUNT_NAME")
        or "aidepartmentstorage"
    )
    blob_container: str = field(
        default_factory=lambda: os.getenv("BLOB_CONTAINER") or "docs"
    )
    blob_name: str = field(
        default_factory=lambda: os.getenv("BLOB_NAME") or "role_library.pdf"
    )

    @property
    def storage_account_url(self) -> str:
        return f"https://{self.storage_account_name}.blob.core.windows.net"

    @property
    def neo4j(self) -> Neo4jSettings:
        return Neo4jSettings(
            uri=_required_env("NEO4J_URI"),
            username=_required_env("NEO4J_USERNAME"),
            password=_required_env("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE") or "neo4j",
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