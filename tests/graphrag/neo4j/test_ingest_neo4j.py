import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from graphrag.Neo4j.ingest_neo4j import load_chunks, main


class LoadChunksTests(unittest.TestCase):
    def test_main_routes_ingestion_through_mcp(self):
        chunks = [{"chunk_id": "one"}]
        with (
            patch("graphrag.Neo4j.ingest_neo4j.load_chunks", return_value=chunks) as load,
            patch("graphrag.Neo4j.ingest_neo4j.DatabaseMcpClient") as factory,
            patch("graphrag.Neo4j.ingest_neo4j.get_chunk_embedding", return_value=[0.1]),
            patch("graphrag.Neo4j.ingest_neo4j.settings") as configuration,
        ):
            configuration.azure_openai_embedding_deployment = "test-embedding"
            client = Mock()
            factory.return_value.__enter__.return_value = client
            main(["--limit", "1"])
        self.assertEqual(load.call_args.args[1], 1)
        client.health_check.assert_called_once()
        client.ensure_schema.assert_called_once()
        client.store_chunk.assert_called_once_with(
            chunks[0], [0.1], embedding_model="test-embedding",
        )

    def test_loads_all_chunk_files_in_stable_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b_chunks.json").write_text(json.dumps([{"chunk_id": "b"}]), encoding="utf-8")
            (root / "a_chunks.json").write_text(json.dumps([{"chunk_id": "a"}]), encoding="utf-8")
            self.assertEqual([chunk["chunk_id"] for chunk in load_chunks(root)], ["a", "b"])

    def test_limit_bounds_smoke_ingestion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [{"chunk_id": str(index)} for index in range(5)]
            (root / "sample_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
            self.assertEqual(len(load_chunks(root, limit=3)), 3)

    def test_missing_files_or_empty_selection_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_chunks(root)
            (root / "sample_chunks.json").write_text(json.dumps([{"chunk_id": "one"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_chunks(root, limit=0)


if __name__ == "__main__":
    unittest.main()