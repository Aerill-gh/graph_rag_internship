
import argparse
import json
from pathlib import Path

from graphrag.config import settings
from graphrag.embeddings import get_chunk_embedding
from graphrag.neo4j_store import (
    create_neo4j_driver,
    create_neo4j_schema,
    store_chunk,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        description="Embed cached chunks and store them in Neo4j."
    )
    parser.add_argument("--limit", type=int, help="Maximum chunks to ingest (use 3 for a smoke test).")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    chunks = load_chunks(CHUNKS_DIR, args.limit)

    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        with driver.session(database=settings.neo4j.database) as session:
            create_neo4j_schema(session)
            for index, chunk in enumerate(chunks, start=1):
                embedding = get_chunk_embedding(chunk)
                store_chunk(session, chunk, embedding)
                print(f"[{index}/{len(chunks)}] Stored: {chunk['chunk_id']}")

    print(f"\nTotal chunks processed: {len(chunks)}")


if __name__ == "__main__":
    main()
