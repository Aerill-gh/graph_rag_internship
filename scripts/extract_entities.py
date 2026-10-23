import json
from pathlib import Path

from graphrag.extract import extract_entities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "chunks" / "role_library_chunks.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "extracted" / "test_extraction.json"


def main():
    chunks = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    if not chunks:
        raise ValueError("No chunks found in the input file.")

    chunk = chunks[0]
    print(f"Extracting: {chunk['chunk_id']}")

    result = extract_entities(chunk)
    output = {
        "chunk_id": chunk["chunk_id"],
        "source_document": chunk["source_document"],
        "pages": chunk.get("pages", []),
        "figures": chunk.get("figures", []),
        "entities": result,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
