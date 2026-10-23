import json
import unittest
from collections import Counter
from pathlib import Path

from graphrag.config import PROJECT_ROOT


class EvaluationQuestionsTests(unittest.TestCase):
    def setUp(self):
        path = PROJECT_ROOT / "evaluation/questions.jsonl"
        self.questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_question_counts_ids_and_required_fields(self):
        self.assertEqual(len(self.questions), 24)
        self.assertEqual(len({question["id"] for question in self.questions}), 24)
        self.assertEqual(Counter(question["type"] for question in self.questions), {
            "simple_fact": 8, "comparison": 6, "list_count": 6, "not_in_documents": 4,
        })
        for question in self.questions:
            self.assertTrue(question["question"].strip())
            self.assertTrue(question["expected_answer"].strip())
            self.assertIsInstance(question["source_document"], list)
            self.assertEqual(bool(question["evidence"]), question["type"] != "not_in_documents")

    def test_evidence_quotes_match_available_source_chunks(self):
        paths = list((PROJECT_ROOT / "data/chunks").glob("*.json"))
        if not paths:
            self.skipTest("Local source chunks are not available.")
        chunks = {
            chunk["chunk_id"]: chunk
            for path in paths for chunk in json.loads(path.read_text(encoding="utf-8"))
        }
        for question in self.questions:
            for evidence in question["evidence"]:
                with self.subTest(question=question["id"], chunk=evidence["chunk_id"]):
                    self.assertIn(evidence["chunk_id"], chunks)
                    chunk = chunks[evidence["chunk_id"]]
                    self.assertIn(chunk["source_document"], question["source_document"])
                    self.assertEqual(evidence["pages"], chunk["pages"])
                    self.assertIn(" ".join(evidence["quote"].split()), " ".join(chunk["text"].split()))


if __name__ == "__main__":
    unittest.main()