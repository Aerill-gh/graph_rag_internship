import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from graphrag.config import FoundrySettings
from graphrag.graph_extraction import extract_chunk


class MockModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, text, instructions, max_output_tokens):
        self.requests.append(text)
        return next(self.responses)


class GraphExtractionTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {
            "chunk_id": "chunk-1",
            "source_document": "benefits.pdf",
            "text": "The Health Plan covers preventive care.",
        }
        self.configuration = FoundrySettings("https://example.test", "chat")
        self.valid = {
            "entities": [
                {"label": "Plan", "name": "Health Plan"},
                {"label": "Service", "name": "preventive care"},
            ],
            "relationships": [{
                "source_label": "Plan", "source_name": "Health Plan",
                "type": "COVERS", "target_label": "Service",
                "target_name": "preventive care",
            }],
        }

    def test_invalid_relationship_endpoint_is_repaired_once(self):
        invalid = {
            **self.valid,
            "relationships": [{
                **self.valid["relationships"][0], "target_name": "wellness care",
            }],
        }
        model = MockModel([json.dumps(invalid), json.dumps(self.valid)])
        with TemporaryDirectory() as directory:
            result = extract_chunk(
                self.chunk, model, Path(directory), self.configuration
            )
            self.assertEqual(result["relationships"], self.valid["relationships"])
            self.assertEqual(len(model.requests), 2)
            self.assertIn("Relationship target is missing", model.requests[1])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)

    def test_still_invalid_response_is_rejected_and_not_cached(self):
        invalid = {
            **self.valid,
            "relationships": [{
                **self.valid["relationships"][0], "target_name": "wellness care",
            }],
        }
        model = MockModel([json.dumps(invalid), json.dumps(invalid)])
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "one repair attempt"):
                extract_chunk(self.chunk, model, Path(directory), self.configuration)
            self.assertEqual(len(model.requests), 2)
            self.assertEqual(list(Path(directory).glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
