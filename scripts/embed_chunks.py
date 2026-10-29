import json
from pathlib import Path

from graphrag.embeddings import get_chunk_embedding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"


def main():
    files = sorted(CHUNKS_DIR.glob("*_chunks.json"))

    if not files:
        raise FileNotFoundError("No chunk JSON files found.")

    total = 0

    for file_path in files:
        with file_path.open(encoding="utf-8") as file:
            chunks = json.load(file)

        print(f"\nProcessing {file_path.name} ({len(chunks)} chunks)")

        for chunk in chunks:
            embedding = get_chunk_embedding(chunk)
            total += 1
            print(
                f"[{total}] Processed: {chunk['chunk_id']} "
                f"({len(embedding)} dimensions)"
            )

    print("\nEmbedding processing completed.")
    print(f"Total chunks processed: {total}")


if __name__ == "__main__":
    main()
