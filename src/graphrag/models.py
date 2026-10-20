
from openai import OpenAI

from graphrag.config import settings


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding using the Azure deployment."""
    if not text.strip():
        raise ValueError("Cannot generate an embedding for empty text.")

    azure_settings = settings.azure_openai

    client = OpenAI(
        api_key=azure_settings.api_key,
        base_url=azure_settings.base_url,
    )

    response = client.embeddings.create(
        model=settings.embedding_deployment,
        input=text,
    )

    return response.data[0].embedding
