import argparse
import json
from pathlib import Path
from typing import Callable

from graphrag.config import PROJECT_ROOT, settings
from graphrag.graph_extraction import extract_chunk, merge_duplicate_entities
from graphrag.neo4j_store import (
    create_neo4j_driver,
    create_neo4j_schema,
    store_chunk,
    store_extraction,
)
from graphrag.embeddings import get_chunk_embedding


DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data/chunks"


def build_cached_graph(session, chunks: list[dict], report_path: Path) -> dict:
    records = []
    missing = []
    for chunk in chunks:
        try:
            records.append(extract_chunk(chunk, cache_only=True))
        except FileNotFoundError:
            missing.append(chunk["chunk_id"])
    merged = merge_duplicate_entities(records)
    for chunk in chunks:
        store_chunk(session, chunk)
    for record in merged:
        store_extraction(session, record["chunk_id"], {
            "entities": record["entities"], "relationships": record["relationships"],
        })
    report = {
        "mode": "cache-only", "chunks": len(chunks),
        "extracted_chunks": len(merged), "missing_chunk_ids": missing,
        "entity_mentions": sum(len(record["entities"]) for record in merged),
        "relationship_mentions": sum(len(record["relationships"]) for record in merged),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_name("offline_extractions.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_chunks(chunks_dir: Path, limit: int | None = None) -> list[dict]:
    files = sorted(chunks_dir.glob("*_chunks.json"))
    if not files:
        raise FileNotFoundError(f"No *_chunks.json files found in {chunks_dir}.")
    chunks = []
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Chunk file must contain a JSON list: {path}")
        chunks.extend(records)
    if limit is not None:
        chunks = chunks[:limit]
    if not chunks:
        raise ValueError("No chunks selected for graph building.")
    ids = [chunk.get("chunk_id") for chunk in chunks]
    if any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in ids):
        raise ValueError("Every chunk must have a nonempty chunk_id.")
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs must be unique across the selected files.")
    return chunks


def build_graph_for_chunks(
    session,
    chunks: list[dict],
    embedding_fn: Callable[[dict], list[float]] = get_chunk_embedding,
    extraction_fn: Callable[[dict], dict] = extract_chunk,
) -> dict[str, int]:
    """Store chunks first, then merge and persist validated graph facts."""
    extraction_records = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {index}/{len(chunks)}: {chunk['chunk_id']}")
        vector = embedding_fn(chunk)
        store_chunk(session, chunk, vector)
        extraction_records.append(extraction_fn(chunk))

    merged_records = merge_duplicate_entities(extraction_records)
    for record in merged_records:
        store_extraction(
            session,
            record["chunk_id"],
            {"entities": record["entities"], "relationships": record["relationships"]},
        )

    return {
        "chunks": len(chunks),
        "entity_mentions": sum(len(record["entities"]) for record in merged_records),
        "relationship_mentions": sum(len(record["relationships"]) for record in merged_records),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Neo4j chunk vector index and ontology graph from chunk JSON files."
    )
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--cache-only", action="store_true", help="Use cached facts only; never call Azure or generate embeddings.")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "data/cache/offline_build_report.json")
    parser.add_argument(
        "--limit", type=int,
        help="Maximum chunks to process. Start with --limit 2; omit to process all chunks.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    chunks = load_chunks(args.chunks_dir, args.limit)

    with create_neo4j_driver() as driver:
        driver.verify_connectivity()
        with driver.session(database=settings.neo4j.database) as session:
            create_neo4j_schema(session)
            if args.cache_only:
                counts = build_cached_graph(session, chunks, args.report)
                print(f"Offline build: {counts['chunks']} source chunks; {counts['extracted_chunks']} cached extractions; {len(counts['missing_chunk_ids'])} missing. Report: {args.report}")
                return 0
            counts = build_graph_for_chunks(session, chunks)

    print(
        "Graph build complete: "
        f"{counts['chunks']} chunks, {counts['entity_mentions']} entity mentions, "
        f"{counts['relationship_mentions']} domain relationship mentions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
