"""Opt-in MCP integration test using two real cached Zava chunks, without Azure calls."""

import json
import os
import unittest
from pathlib import Path

from graphrag.database_mcp.client import DatabaseMcpClient
from graphrag.graph_extraction import extract_chunk, merge_duplicate_entities


@unittest.skipUnless(
    os.getenv("RUN_DATABASE_MCP_LIVE_TESTS") == "true",
    "Set RUN_DATABASE_MCP_LIVE_TESTS=true after starting the database MCP server.",
)
class LiveDatabaseMcpTests(unittest.TestCase):
    def test_cached_sources_facts_and_repeat_ingestion(self):
        root = Path(__file__).resolve().parents[2]
        chunks = [
            chunk
            for path in sorted((root / "data" / "chunks").glob("*_chunks.json"))
            for chunk in json.loads(path.read_text(encoding="utf-8"))
        ][:2]
        self.assertEqual(len(chunks), 2, "Two cached Zava chunks are required.")
        records = merge_duplicate_entities([
            extract_chunk(chunk, cache_only=True) for chunk in chunks
        ])

        def write(client):
            for chunk in chunks:
                client.store_chunk(chunk)
            for record in records:
                client.store_extraction(record["chunk_id"], {
                    "entities": record["entities"], "relationships": record["relationships"],
                })

        with DatabaseMcpClient() as client:
            client.health_check()
            client.ensure_schema()
            write(client)
            first = [client.get_chunk(chunk["chunk_id"]) for chunk in chunks]
            for chunk, snapshot in zip(chunks, first):
                self.assertEqual(snapshot.chunk.model_dump(), {
                    **chunk, "figures": chunk.get("figures", []),
                })
                self.assertFalse(snapshot.entities_truncated)
                self.assertFalse(snapshot.relationships_truncated)
                record = next(record for record in records if record["chunk_id"] == chunk["chunk_id"])
                for entity in record["entities"]:
                    self.assertIn(entity, [item.model_dump() for item in snapshot.entities])
                for relationship in record["relationships"]:
                    self.assertIn(
                        relationship, [item.model_dump() for item in snapshot.relationships],
                    )
            write(client)
            second = [client.get_chunk(chunk["chunk_id"]) for chunk in chunks]
            self.assertEqual(first, second, "Repeat ingestion must preserve vectors and avoid duplicates.")


if __name__ == "__main__":
    unittest.main()
