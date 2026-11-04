import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

from graphrag.build_graph import build_cached_graph, build_graph_for_chunks, load_chunks, main


class BuildGraphTests(unittest.TestCase):
    def test_cache_only_main_uses_mcp_without_running_embedding_build(self):
        chunks = [{"chunk_id": "one"}]
        with (
            patch("graphrag.build_graph.load_chunks", return_value=chunks),
            patch("graphrag.build_graph.DatabaseMcpClient") as factory,
            patch("graphrag.build_graph.build_cached_graph", return_value={
                "chunks": 1, "extracted_chunks": 1, "missing_chunk_ids": [],
            }) as cached,
            patch("graphrag.build_graph.build_graph_for_chunks") as embedding_build,
        ):
            client = Mock()
            factory.return_value.__enter__.return_value = client
            self.assertEqual(main(["--cache-only", "--limit", "1"]), 0)
        client.health_check.assert_called_once()
        client.ensure_schema.assert_called_once()
        self.assertEqual(cached.call_args.args[:2], (client, chunks))
        embedding_build.assert_not_called()

    def test_offline_build_reports_missing_and_stores_available_facts(self):
        chunks = [{"chunk_id": "one"}, {"chunk_id": "two"}]
        record = {"chunk_id": "one", "entities": [{"label": "Role", "name": "Engineer"}], "relationships": []}
        with TemporaryDirectory() as directory, patch(
            "graphrag.build_graph.extract_chunk", side_effect=[record, FileNotFoundError()]
        ) as extract:
            client = Mock()
            report_path = Path(directory) / "report.json"
            report = build_cached_graph(client, chunks, report_path)
            self.assertEqual(report["missing_chunk_ids"], ["two"])
            self.assertEqual(report["extracted_chunks"], 1)
            self.assertEqual(client.store_chunk.call_count, 2)
            self.assertEqual(client.store_extraction.call_count, 1)
            self.assertTrue(all(item.kwargs == {"cache_only": True} for item in extract.call_args_list))
            self.assertEqual(json.loads(report_path.read_text()), report)
            self.assertEqual(json.loads(report_path.with_name("offline_extractions.json").read_text()), [record])

    def test_load_chunks_is_sorted_and_supports_a_pilot_limit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b_chunks.json").write_text(json.dumps([{"chunk_id": "b"}]), encoding="utf-8")
            (root / "a_chunks.json").write_text(json.dumps([{"chunk_id": "a"}, {"chunk_id": "a2"}]), encoding="utf-8")
            self.assertEqual([chunk["chunk_id"] for chunk in load_chunks(root)], ["a", "a2", "b"])
            self.assertEqual([chunk["chunk_id"] for chunk in load_chunks(root, 2)], ["a", "a2"])

    def test_build_stores_chunk_vectors_then_merged_graph_facts(self):
        chunks = [
            {"chunk_id": "one", "source_document": "roles.pdf", "text": "Engineer needs Python."},
            {"chunk_id": "two", "source_document": "roles.pdf", "text": "engineer needs python."},
        ]
        extracted = [
            {"chunk_id": "one", "entities": [
                {"label": "Role", "name": "Engineer"}, {"label": "Skill", "name": "Python"},
            ], "relationships": [{
                "source_label": "Role", "source_name": "Engineer", "type": "REQUIRES",
                "target_label": "Skill", "target_name": "Python",
            }]},
            {"chunk_id": "two", "entities": [
                {"label": "Role", "name": " engineer "}, {"label": "Skill", "name": "python"},
            ], "relationships": [{
                "source_label": "Role", "source_name": " engineer ", "type": "REQUIRES",
                "target_label": "Skill", "target_name": "python",
            }]},
        ]
        embed = lambda chunk: [float(len(chunk["text"]))]
        extract = lambda chunk: extracted[0 if chunk["chunk_id"] == "one" else 1]
        client = Mock()
        with patch("graphrag.build_graph.settings") as configuration:
            configuration.azure_openai_embedding_deployment = "test-embedding"
            counts = build_graph_for_chunks(client, chunks, embed, extract)

        self.assertEqual(counts, {"chunks": 2, "entity_mentions": 4, "relationship_mentions": 2})
        self.assertEqual(client.store_chunk.call_args_list, [
            call(chunks[0], [len(chunks[0]["text"])], embedding_model="test-embedding"),
            call(chunks[1], [len(chunks[1]["text"])], embedding_model="test-embedding"),
        ])
        self.assertEqual(client.store_extraction.call_count, 2)
        first_record = client.store_extraction.call_args_list[0].args[1]
        self.assertEqual(first_record["entities"], [
            {"label": "Role", "name": "Engineer"}, {"label": "Skill", "name": "Python"},
        ])
        self.assertEqual(first_record["relationships"][0]["source_name"], "Engineer")

    def test_write_failure_stops_build(self):
        client = Mock()
        client.store_chunk.side_effect = RuntimeError("MCP write failed")
        extraction = Mock()
        with patch("graphrag.build_graph.settings") as configuration:
            configuration.azure_openai_embedding_deployment = "test-embedding"
            with self.assertRaisesRegex(RuntimeError, "MCP write failed"):
                build_graph_for_chunks(
                    client, [{"chunk_id": "one"}], lambda chunk: [0.1], extraction,
                )
        extraction.assert_not_called()
        client.store_extraction.assert_not_called()

    def test_duplicate_chunk_ids_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunks_chunks.json").write_text(json.dumps([
                {"chunk_id": "same"}, {"chunk_id": "same"},
            ]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_chunks(root)


if __name__ == "__main__":
    unittest.main()
