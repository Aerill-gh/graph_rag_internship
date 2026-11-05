
import argparse
import json
from pathlib import Path

from graphrag.database_mcp.client import DatabaseMcpClient
from graphrag.config import PROJECT_ROOT, settings
from graphrag.embeddings import get_chunk_embedding


CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"


def load_chunks(chunks_dir: Path, limit: int | None = None) -> list[dict]:
    files = sorted(chunks_dir.glob("*_chunks.json"))

    if not files:
        raise FileNotFoundError("No chunk JSON files found.")

    chunks = []
    for file_path in files:
        chunks.extend(json.loads(file_path.read_text(encoding="utf-8")))
    if limit is not None:
        chunks = chunks[:limit]
    if not chunks:
        raise ValueError("No chunks selected for ingestion.")
    return chunks


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Embed cached chunks and store them through database MCP."
    )
    parser.add_argument("--limit", type=int, help="Maximum chunks to ingest (use 3 for a smoke test).")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    chunks = load_chunks(CHUNKS_DIR, args.limit)

    with DatabaseMcpClient() as client:
        client.health_check()
        client.ensure_schema()
        for index, chunk in enumerate(chunks, start=1):
            embedding = get_chunk_embedding(chunk)
            client.store_chunk(
                chunk, embedding, embedding_model=settings.azure_openai_embedding_deployment,
            )
            print(f"[{index}/{len(chunks)}] Stored: {chunk['chunk_id']}")

    print(f"\nTotal chunks processed: {len(chunks)}")


if __name__ == "__main__":
    main()
