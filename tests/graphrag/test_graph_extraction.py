import json
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from graphrag.config import FoundrySettings
from graphrag.graph_extraction import EXTRACTION_INSTRUCTIONS, extract_chunk


class MockModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, text, instructions, max_output_tokens):
        self.requests.append(text)
        return next(self.responses)


class GraphExtractionTests(unittest.TestCase):
    def test_extraction_instructions_do_not_infer_exclusions_from_missing_items(self):
        self.assertIn(
            "Do not infer an EXCLUDES relationship merely because a service is absent",
            EXTRACTION_INSTRUCTIONS,
        )
        self.assertIn(
            "do not mark dental, dental exams,",
            EXTRACTION_INSTRUCTIONS,
        )

    def test_reviewed_shared_dental_coverage_removes_inferred_exclusions(self):
        chunk = {
            "chunk_id": "Benefit_Options_002",
            "source_document": "Benefit_Options.pdf",
            "text": "Both plans offer coverage for vision and dental services.",
        }
        extraction = {
            "entities": [
                {"label": "Plan", "name": "Northwind Standard"},
                {"label": "Service", "name": "dental"},
                {"label": "Service", "name": "dental services"},
                {"label": "Service", "name": "dental exams"},
                {"label": "Service", "name": "cleanings"},
                {"label": "Service", "name": "fillings"},
                {"label": "Service", "name": "emergency services"},
            ],
            "relationships": [
                {"source_label": "Plan", "source_name": "Northwind Standard", "type": "EXCLUDES",
                 "target_label": "Service", "target_name": "dental"},
                {"source_label": "Plan", "source_name": "Northwind Standard", "type": "EXCLUDES",
                 "target_label": "Service", "target_name": "dental services"},
                {"source_label": "Plan", "source_name": "Northwind Standard", "type": "EXCLUDES",
                 "target_label": "Service", "target_name": "dental exams"},
                {"source_label": "Plan", "source_name": "Northwind Standard", "type": "EXCLUDES",
                 "target_label": "Service", "target_name": "cleanings"},
                {"source_label": "Plan", "source_name": "Northwind Standard", "type": "EXCLUDES",
                 "target_label": "Service", "target_name": "fillings"},
                {"source_label": "Plan", "source_name": "Northwind Standard", "type": "EXCLUDES",
                 "target_label": "Service", "target_name": "emergency services"},
            ],
        }
        model = MockModel([json.dumps(extraction)])
        with TemporaryDirectory() as directory:
            result = extract_chunk(chunk, model, Path(directory), self.configuration)

        self.assertEqual(
            [relationship["target_name"] for relationship in result["relationships"]],
            ["emergency services"],
        )

    def test_explicit_dental_exclusion_is_preserved(self):
        chunk = {
            "chunk_id": "Benefit_Options_002",
            "source_document": "Benefit_Options.pdf",
            "text": "Both plans offer coverage for vision and dental services. Northwind Standard does not cover fillings.",
        }
        extraction = {
            "entities": [
                {"label": "Plan", "name": "Northwind Standard"},
                {"label": "Service", "name": "fillings"},
            ],
            "relationships": [{
                "source_label": "Plan", "source_name": "Northwind Standard", "type": "EXCLUDES",
                "target_label": "Service", "target_name": "fillings",
            }],
        }
        model = MockModel([json.dumps(extraction)])
        with TemporaryDirectory() as directory:
            result = extract_chunk(chunk, model, Path(directory), self.configuration)

        self.assertEqual(result["relationships"], extraction["relationships"])

    def test_cache_only_miss_never_constructs_model(self):
        with TemporaryDirectory() as directory, patch(
            "graphrag.graph_extraction.FoundryExtractionModel"
        ) as model:
            with self.assertRaises(FileNotFoundError):
                extract_chunk(self.chunk, cache_dir=Path(directory),
                              configuration=self.configuration, cache_only=True)
            model.assert_not_called()

    def test_cache_only_hit_never_constructs_model(self):
        with TemporaryDirectory() as directory:
            extract_chunk(self.chunk, MockModel([json.dumps(self.valid)]),
                          Path(directory), self.configuration)
            with patch("graphrag.graph_extraction.FoundryExtractionModel") as model:
                result = extract_chunk(self.chunk, cache_dir=Path(directory),
                                       configuration=self.configuration, cache_only=True)
                self.assertEqual(result["relationships"], self.valid["relationships"])
                model.assert_not_called()

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

    def test_company_exclusion_repair_is_guided_to_ontology_pattern(self):
        invalid = {
            "entities": [
                {"label": "Company", "name": "Northwind Health Plus"},
                {"label": "Service", "name": "home health care"},
            ],
            "relationships": [{
                "source_label": "Company", "source_name": "Northwind Health Plus",
                "type": "EXCLUDES", "target_label": "Service", "target_name": "home health care",
            }],
        }
        corrected = {
            "entities": [
                {"label": "Plan", "name": "Northwind Health Plus"},
                {"label": "Service", "name": "home health care"},
            ],
            "relationships": [{
                "source_label": "Plan", "source_name": "Northwind Health Plus",
                "type": "EXCLUDES", "target_label": "Service", "target_name": "home health care",
            }],
        }
        model = MockModel([json.dumps(invalid), json.dumps(corrected)])
        with TemporaryDirectory() as directory:
            result = extract_chunk(
                self.chunk, model, Path(directory), self.configuration
            )

        self.assertEqual(result["relationships"], corrected["relationships"])
        self.assertIn("A Company cannot directly COVERS or EXCLUDES a Service", model.requests[1])
        self.assertEqual(len(model.requests), 2)

    def test_service_covers_service_repair_uses_explicit_plan(self):
        chunk = {
            "chunk_id": "chunk-1",
            "source_document": "benefits.pdf",
            "text": "The Northwind Health Plus plan covers a Chronic Condition Management Program.",
        }
        invalid = {
            "entities": [
                {"label": "Service", "name": "Disease Management"},
                {"label": "Service", "name": "Chronic Condition Management Program"},
            ],
            "relationships": [{
                "source_label": "Service", "source_name": "Disease Management",
                "type": "COVERS", "target_label": "Service",
                "target_name": "Chronic Condition Management Program",
            }],
        }
        corrected = {
            "entities": [
                {"label": "Plan", "name": "Northwind Health Plus"},
                {"label": "Service", "name": "Chronic Condition Management Program"},
            ],
            "relationships": [{
                "source_label": "Plan", "source_name": "Northwind Health Plus",
                "type": "COVERS", "target_label": "Service",
                "target_name": "Chronic Condition Management Program",
            }],
        }
        model = MockModel([json.dumps(invalid), json.dumps(corrected)])
        with TemporaryDirectory() as directory:
            result = extract_chunk(
                chunk, model, Path(directory), self.configuration
            )

        self.assertEqual(result["relationships"], corrected["relationships"])
        self.assertIn("Never create relationships from one Service to", model.requests[1])
        self.assertEqual(len(model.requests), 2)

    def test_persistent_unsupported_relationship_is_skipped_and_reported(self):
        invalid = {
            "entities": [
                {"label": "Service", "name": "Disease Management"},
                {"label": "Service", "name": "Chronic Condition Management Program"},
            ],
            "relationships": [{
                "source_label": "Service", "source_name": "Disease Management",
                "type": "COVERS", "target_label": "Service",
                "target_name": "Chronic Condition Management Program",
            }],
        }
        model = MockModel([json.dumps(invalid), json.dumps(invalid)])
        with TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                result = extract_chunk(
                    self.chunk, model, Path(directory), self.configuration
                )

            self.assertEqual(result["relationships"], [])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)
        self.assertIn("Skipped 1 ontology-invalid relationship(s) in chunk-1", output.getvalue())
        self.assertEqual(len(model.requests), 2)

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

    def test_company_exclusion_repair_uses_supported_plan_rule(self):
        invalid = {
            "entities": [
                {"label": "Company", "name": "Health Plan"},
                {"label": "Service", "name": "cosmetic procedures"},
            ],
            "relationships": [{
                "source_label": "Company", "source_name": "Health Plan",
                "type": "EXCLUDES", "target_label": "Service",
                "target_name": "cosmetic procedures",
            }],
        }
        corrected = {
            "entities": [
                {"label": "Plan", "name": "Health Plan"},
                {"label": "Service", "name": "cosmetic procedures"},
            ],
            "relationships": [{
                **invalid["relationships"][0], "source_label": "Plan",
            }],
        }
        chunk = {**self.chunk, "text": "The Health Plan does not cover cosmetic procedures."}
        model = MockModel([json.dumps(invalid), json.dumps(corrected)])
        with TemporaryDirectory() as directory:
            result = extract_chunk(chunk, model, Path(directory), self.configuration)
            self.assertEqual(result["entities"], corrected["entities"])
            self.assertEqual(result["relationships"], corrected["relationships"])
            self.assertIn("Unsupported relationship pattern:", model.requests[1])
            self.assertIn("COVERS and EXCLUDES must start from a Plan", model.requests[1])
            self.assertIn("Do not guess a label to satisfy validation", model.requests[1])
            cached = extract_chunk(
                chunk, cache_dir=Path(directory),
                configuration=self.configuration, cache_only=True,
            )
            self.assertEqual(cached, result)


if __name__ == "__main__":
    unittest.main()
