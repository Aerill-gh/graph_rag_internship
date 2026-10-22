import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from graphrag.chatbot.agent import AgentAnswer
from scripts.ask import main


class AskCliTests(unittest.TestCase):
    def setUp(self):
        agent_patch = patch("scripts.ask.BaselineAgent")
        self.agent_class = agent_patch.start()
        self.addCleanup(agent_patch.stop)
        self.agent_class.return_value.ask.return_value = AgentAnswer(
            "Engineer requires Python [S1].",
            [{"citation_id": "S1", "source_document": "role_library.pdf",
              "pages": [2], "chunk_id": "role_library_001"}],
            1,
        )

    def test_text_output_prints_source(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["Which role requires Python?"]), 0)
        self.assertIn("Engineer requires Python [S1].", output.getvalue())
        self.assertIn("role_library.pdf; pages 2", output.getvalue())

    def test_json_output_is_machine_readable(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["Question", "--json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["sources"][0]["citation_id"], "S1")

    def test_service_errors_do_not_print_exception_details(self):
        self.agent_class.return_value.ask.side_effect = Exception("secret-value")
        output = io.StringIO()
        with redirect_stderr(output):
            self.assertEqual(main(["Question"]), 1)
        self.assertNotIn("secret-value", output.getvalue())

    def test_blank_question_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([" "])
        self.agent_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()