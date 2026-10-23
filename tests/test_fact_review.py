import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.sample_graph_facts import build_review_rows, main


class FactReviewTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {
            "chunk_id": "chunk-1", "source_document": "roles.pdf", "pages": [1, 2],
            "text": "Engineer requires Python.",
        }
        self.extractions = [{
            "chunk_id": "chunk-1",
            "entities": [{"label": "Role", "name": "Engineer"}, {"label": "Skill", "name": "Python"}],
            "relationships": [{"source_label": "Role", "source_name": "Engineer", "type": "REQUIRES",
                               "target_label": "Skill", "target_name": "Python"}],
        }]

    def test_review_rows_carry_fact_and_source_provenance(self):
        rows = build_review_rows(self.extractions, {"chunk-1": self.chunk})
        self.assertEqual(len(rows), 3)
        relation = rows[-1]
        self.assertIn("Role 'Engineer' -REQUIRES-> Skill 'Python'", relation["fact"])
        self.assertEqual(relation["source_text"], self.chunk["text"])
        self.assertEqual(relation["pages"], [1, 2])
        self.assertIsNone(relation["verified"])

    def test_missing_chunk_and_unapproved_fact_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing source chunk"):
            build_review_rows(self.extractions, {})
        invalid = [{"chunk_id": "chunk-1", "entities": [{"label": "Employee", "name": "A"}],
                    "relationships": []}]
        with self.assertRaises(ValueError):
            build_review_rows(invalid, {"chunk-1": self.chunk})

    def test_missing_extraction_file_reports_required_first_step(self):
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            output = io.StringIO()
            with redirect_stderr(output), self.assertRaises(SystemExit) as error:
                main(["--extractions", str(missing)])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Extraction output is missing", output.getvalue())
            self.assertIn("extract_graph_entities.py --limit 2", output.getvalue())
            self.assertNotIn("Traceback", output.getvalue())

    def test_sample_output_is_seeded_jsonl_and_size_is_bounded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            chunks_dir = root / "chunks"
            chunks_dir.mkdir()
            (chunks_dir / "chunks.json").write_text(json.dumps([self.chunk]), encoding="utf-8")
            extraction_path = root / "extractions.json"
            extraction_path.write_text(json.dumps(self.extractions), encoding="utf-8")
            first, second = root / "first.jsonl", root / "second.jsonl"
            self.assertIsNone(main([
                "--extractions", str(extraction_path), "--chunks-dir", str(chunks_dir),
                "--output", str(first), "--sample-size", "2", "--seed", "10",
            ]))
            main([
                "--extractions", str(extraction_path), "--chunks-dir", str(chunks_dir),
                "--output", str(second), "--sample-size", "2", "--seed", "10",
            ])
            self.assertEqual(first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8"))
            self.assertEqual(len(first.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()