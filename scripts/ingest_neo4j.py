
import json
from pathlib import Path

from graphrag.embeddings import get_chunk_embedding
from graphrag.neo4j_store import create_neo4j_driver, store_chunk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"


def main():
    files = sorted(CHUNKS_DIR.glob("*_chunks.json"))

    if not files:
        raise FileNotFoundError("No chunk JSON files found.")

    total = 0

    with create_neo4j_driver() as driver:
        driver.verify_connectivity()

        with driver.session(database="neo4j") as session:
            for file_path in files:
                chunks = json.loads(
                    file_path.read_text(encoding="utf-8")
                )

                print(f"Processing {file_path.name}: {len(chunks)} chunks")

                for chunk in chunks:
                    embedding = get_chunk_embedding(chunk)
                    store_chunk(session, chunk, embedding)
                    total += 1

                print(f"Completed: {file_path.name}")

    print(f"\nTotal chunks processed: {total}")


if __name__ == "__main__":
    main()
