import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from graphrag.config import PROJECT_ROOT
from graphrag.chatbot.agent import AgentAnswer
from evaluation.run_eval import run_evaluation, comparison_summary


class EvaluationQuestionsTests(unittest.TestCase):
    def test_explicit_error_retry_keeps_attempt_history(self):
        agents = {mode: Mock() for mode in ("baseline", "graphrag")}
        agents["baseline"].ask.side_effect = [RuntimeError("network"), AgentAnswer("No evidence.", [], 1)]
        agents["graphrag"].ask.return_value = AgentAnswer("No evidence.", [], 1)
        question = {"id": "one", "question": "Question", "expected_answer": "Answer", "type": "simple_fact"}
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_evaluation([question], output, agents)
            rows = run_evaluation([question], output, agents, resume=True, retry_errors=True)
            self.assertEqual(len((output / "answers.jsonl").read_text().splitlines()), 2)
        retried = next(row for row in rows if row["mode"] == "baseline")
        self.assertNotIn("error", retried)
        self.assertEqual(retried["previous_attempts"][0]["error"], "RuntimeError")
        agents["graphrag"].ask.assert_called_once()

    def test_resume_does_not_repeat_completed_answers(self):
        agents = {mode: Mock() for mode in ("baseline", "graphrag")}
        for agent in agents.values():
            agent.ask.return_value = AgentAnswer("No evidence.", [], 1, ["search_documents"])
        question = {"id": "one", "question": "Question", "expected_answer": "Answer", "type": "simple_fact"}
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            self.assertEqual(len(run_evaluation([question], output, agents, max_new_answers=1)), 1)
            rows = run_evaluation([question], output, agents, resume=True, workers=2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(run_evaluation([question], output, agents, resume=True)), 2)
        for agent in agents.values():
            agent.ask.assert_called_once()

    def test_comparison_keeps_both_modes_and_manual_scores_separate(self):
        agents = {mode: Mock() for mode in ("baseline", "graphrag")}
        agents["baseline"].ask.return_value = AgentAnswer("Answer [S1].", [{"chunk_id": "one"}], 1, ["search_documents"])
        agents["graphrag"].ask.return_value = AgentAnswer("Answer [S1].", [{"chunk_id": "one"}], 1, ["query_graph"])
        question = {"id": "one", "question": "Question", "expected_answer": "Answer", "type": "comparison", "evidence": [{"chunk_id": "one"}]}
        judge = Mock(return_value={"answer_score": 2, "sources_supported": True, "tool_choice_sensible": True, "reason": "Supported."})
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            rows = run_evaluation([question], output, agents, judge)
            self.assertEqual(len((output / "answers.jsonl").read_text().splitlines()), 2)
            self.assertTrue((output / "summary.json").is_file())
            with self.assertRaises(FileExistsError):
                run_evaluation([question], output, agents, judge)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["manual_grade"]["answer_score"] is None for row in rows))
        self.assertEqual(comparison_summary(rows)["graphrag"]["comparison"]["graph_tool_questions"], 1)
        for agent in agents.values():
            agent.ask.assert_called_once_with("Question")

    def test_service_error_is_recorded_without_secret_details(self):
        agents = {mode: Mock() for mode in ("baseline", "graphrag")}
        for agent in agents.values():
            agent.ask.side_effect = RuntimeError("secret-value")
        question = {"id": "one", "question": "Question", "expected_answer": "Answer", "type": "simple_fact"}
        with TemporaryDirectory() as temporary:
            rows = run_evaluation([question], Path(temporary) / "run", agents)
        self.assertTrue(all(row["error"] == "RuntimeError" for row in rows))
        self.assertNotIn("secret-value", json.dumps(rows))

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