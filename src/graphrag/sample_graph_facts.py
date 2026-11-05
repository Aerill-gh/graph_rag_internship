import argparse
import json
import random
from pathlib import Path

from graphrag.config import PROJECT_ROOT
from graphrag.graph_extraction import validate_extraction


DEFAULT_EXTRACTIONS = PROJECT_ROOT / "data/cache/graph_extraction/extracted_entities.json"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data/chunks"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/cache/graph_extraction/fact_review.jsonl"


def build_review_rows(extractions: list[dict], chunks: dict[str, dict]) -> list[dict]:
    rows = []
    for record in extractions:
        validate_extraction({
            "entities": record.get("entities"),
            "relationships": record.get("relationships"),
        })
        chunk_id = record["chunk_id"]
        if chunk_id not in chunks:
            raise ValueError(f"Missing source chunk for extraction {chunk_id}.")
        chunk = chunks[chunk_id]
        base = {
            "chunk_id": chunk_id,
            "source_document": chunk["source_document"],
            "pages": chunk.get("pages", []),
            "source_text": chunk["text"],
            "verified": None,
            "reviewer_notes": "",
        }
        for entity in record["entities"]:
            rows.append({
                **base, "fact_type": "entity",
                "fact": f"The chunk explicitly mentions {entity['label']} '{entity['name']}'.",
            })
        for relation in record["relationships"]:
            rows.append({
                **base, "fact_type": "relationship",
                "fact": (
                    f"{relation['source_label']} '{relation['source_name']}' "
                    f"-{relation['type']}-> {relation['target_label']} '{relation['target_name']}'."
                ),
            })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a seeded, source-linked manual fact-check sample."
    )
    parser.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive.")
    if not args.extractions.is_file():
        parser.error(
            "Extraction output is missing. First run "
            "'uv run python scripts/extract_graph_entities.py --limit 2' successfully, "
            "or supply an existing file with --extractions."
        )
    extractions = json.loads(args.extractions.read_text(encoding="utf-8"))
    if not isinstance(extractions, list):
        raise ValueError("Extractions file must contain a JSON list.")
    chunks = {}
    for path in sorted(args.chunks_dir.glob("*.json")):
        for chunk in json.loads(path.read_text(encoding="utf-8")):
            if chunk["chunk_id"] in chunks:
                raise ValueError(f"Duplicate source chunk ID: {chunk['chunk_id']}")
            chunks[chunk["chunk_id"]] = chunk
    rows = build_review_rows(extractions, chunks)
    if not rows:
        raise ValueError("No extracted facts are available to review.")
    selected = random.Random(args.seed).sample(rows, min(args.sample_size, len(rows)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(f"Created {len(selected)} pending fact checks at {args.output}.")
    print("Review each fact against source_text and the original document; mark verified true or false.")


if __name__ == "__main__":
    main()