import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

from scripts.build_graph import build_graph_for_chunks, load_chunks


class BuildGraphTests(unittest.TestCase):
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
        with patch("scripts.build_graph.store_chunk") as store_chunk_mock, patch(
            "scripts.build_graph.store_extraction"
        ) as store_extraction_mock:
            counts = build_graph_for_chunks(MockSession(), chunks, embed, extract)

        self.assertEqual(counts, {"chunks": 2, "entity_mentions": 4, "relationship_mentions": 2})
        self.assertEqual(store_chunk_mock.call_args_list, [
            call(unittest.mock.ANY, chunks[0], [len(chunks[0]["text"])]),
            call(unittest.mock.ANY, chunks[1], [len(chunks[1]["text"])])
        ])
        self.assertEqual(store_extraction_mock.call_count, 2)
        first_record = store_extraction_mock.call_args_list[0].args[2]
        self.assertEqual(first_record["entities"], [
            {"label": "Role", "name": "Engineer"}, {"label": "Skill", "name": "Python"},
        ])
        self.assertEqual(first_record["relationships"][0]["source_name"], "Engineer")

    def test_duplicate_chunk_ids_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunks_chunks.json").write_text(json.dumps([
                {"chunk_id": "same"}, {"chunk_id": "same"},
            ]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_chunks(root)


class MockSession:
    pass


if __name__ == "__main__":
    unittest.main()
