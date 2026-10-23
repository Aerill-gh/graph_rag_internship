
import hashlib
import json
from pathlib import Path

from graphrag.config import settings
from graphrag.models import generate_embedding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "embeddings"


def get_chunk_embedding(chunk: dict) -> list[float]:
    """Return an embedding for a chunk, using the existing cache if available."""
    text = chunk["text"]

    if not text.strip():
        raise ValueError(f"Empty text: {chunk['chunk_id']}")

    model = settings.azure_openai_embedding_deployment
    cache_key = hashlib.sha256(
        f"{model}:{text}".encode("utf-8")
    ).hexdigest()

    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        embedding = cached["embedding"]
    else:
        embedding = generate_embedding(text)

    if len(embedding) != 1536:
        raise ValueError(
            f"Unexpected embedding dimensions: {len(embedding)}"
        )

    if not cache_path.is_file():
        result = {
            "chunk_id": chunk["chunk_id"],
            "source_document": chunk["source_document"],
            "pages": chunk["pages"],
            "embedding_model": model,
            "embedding": embedding,
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(result, ensure_ascii=False),
            encoding="utf-8",
        )

    return embedding
