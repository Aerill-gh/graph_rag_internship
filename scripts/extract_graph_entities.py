import argparse
import json
from pathlib import Path

from graphrag.config import PROJECT_ROOT
from graphrag.graph_extraction import extract_chunk, merge_duplicate_entities


DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data/chunks"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/cache/graph_extraction/extracted_entities.json"


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Extract and merge approved graph facts from existing chunk JSON files."
    )
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit", type=int,
        help="Optional maximum number of chunks to process, for a cost-bounded pilot.",
    )
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    chunks = []
    for path in sorted(args.chunks_dir.glob("*.json")):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            raise ValueError(f"Chunk file must contain a JSON list: {path}")
        for chunk in parsed:
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk entries must be JSON objects: {path}")
            chunks.append(chunk)
    if args.limit is not None:
        chunks = chunks[:args.limit]
    if not chunks:
        raise ValueError(f"No chunks found in {args.chunks_dir}.")
    identifiers = [chunk.get("chunk_id") for chunk in chunks]
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers):
        raise ValueError("Every chunk must have a nonempty chunk_id.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Chunk IDs must be unique across the selected files.")

    extracted = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"Extracting chunk {index}/{len(chunks)}: {chunk['chunk_id']}")
        extracted.append(extract_chunk(chunk))
    merged = merge_duplicate_entities(extracted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    entity_count = sum(len(record["entities"]) for record in merged)
    relationship_count = sum(len(record["relationships"]) for record in merged)
    print(f"Processed {len(merged)} chunks; {entity_count} entity mentions and "
          f"{relationship_count} relationship mentions saved to {args.output}.")


if __name__ == "__main__":
    main()