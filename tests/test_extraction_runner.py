import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.extract_graph_entities import main


class ExtractionRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.chunks_dir = self.root / "chunks"
        self.chunks_dir.mkdir()
        self.output = self.root / "merged.json"
        chunks = [
            {"chunk_id": "b_1", "source_document": "b.pdf", "text": "A."},
            {"chunk_id": "a_1", "source_document": "a.pdf", "text": "B."},
        ]
        (self.chunks_dir / "b.json").write_text(json.dumps(chunks[:1]), encoding="utf-8")
        (self.chunks_dir / "a.json").write_text(json.dumps(chunks[1:]), encoding="utf-8")
        self.extraction_patch = patch("scripts.extract_graph_entities.extract_chunk")
        self.merge_patch = patch("scripts.extract_graph_entities.merge_duplicate_entities")
        self.extract = self.extraction_patch.start()
        self.merge = self.merge_patch.start()
        self.addCleanup(self.extraction_patch.stop)
        self.addCleanup(self.merge_patch.stop)
        self.extract.side_effect = lambda chunk: {
            "chunk_id": chunk["chunk_id"], "entities": [], "relationships": [],
        }
        self.merge.side_effect = lambda records: records

    def test_runner_processes_every_sorted_chunk_by_default(self):
        main(["--chunks-dir", str(self.chunks_dir), "--output", str(self.output)])
        self.assertEqual([call.args[0]["chunk_id"] for call in self.extract.call_args_list], ["a_1", "b_1"])
        self.assertEqual(len(json.loads(self.output.read_text(encoding="utf-8"))), 2)

    def test_limit_bounds_extraction_calls(self):
        main(["--chunks-dir", str(self.chunks_dir), "--output", str(self.output), "--limit", "1"])
        self.extract.assert_called_once()
        self.assertEqual(self.extract.call_args.args[0]["chunk_id"], "a_1")

    def test_duplicate_chunk_ids_across_files_are_rejected_before_llm_calls(self):
        (self.chunks_dir / "b.json").write_text(json.dumps([{
            "chunk_id": "a_1", "source_document": "b.pdf", "text": "duplicate",
        }]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique"):
            main(["--chunks-dir", str(self.chunks_dir), "--output", str(self.output)])
        self.extract.assert_not_called()

    def test_empty_corpus_and_bad_limit_are_rejected(self):
        with TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ValueError, "No chunks"):
                main(["--chunks-dir", empty, "--output", str(self.output)])
        with self.assertRaises(SystemExit):
            main(["--chunks-dir", str(self.chunks_dir), "--output", str(self.output), "--limit", "0"])


if __name__ == "__main__":
    unittest.main()