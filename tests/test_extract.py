import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from graphrag.config import FoundrySettings
from graphrag.graph_extraction import (
    FoundryExtractionModel,
    extract_chunk,
    merge_duplicate_entities,
    validate_extraction,
)


CONFIGURATION = FoundrySettings("https://example.test/project", "chat")


def record(chunk_id, entities, relationships):
    return {"chunk_id": chunk_id, "entities": entities, "relationships": relationships}


class ExtractionTests(unittest.TestCase):
    def test_json_mode_requirement_is_in_response_input_not_only_instructions(self):
        with (
            patch("graphrag.graph_extraction.DefaultAzureCredential"),
            patch("graphrag.graph_extraction.AIProjectClient") as project_class,
        ):
            project = project_class.return_value.__enter__.return_value
            client = project.get_openai_client.return_value.__enter__.return_value
            client.responses.create.return_value.status = "completed"
            client.responses.create.return_value.output_text = '{"entities":[],"relationships":[]}'
            source_data = '{"text":"Engineer requires Python."}'

            result = FoundryExtractionModel(CONFIGURATION).chat(
                source_data, "Extract explicit facts.", max_output_tokens=3000,
            )

            self.assertEqual(json.loads(result), {"entities": [], "relationships": []})
            parameters = client.responses.create.call_args.kwargs
            self.assertIn("json", parameters["input"].lower())
            self.assertTrue(parameters["input"].endswith(source_data))
            self.assertEqual(parameters["text"], {"format": {"type": "json_object"}})
            self.assertEqual(parameters["model"], CONFIGURATION.chat_deployment)

    def test_all_approved_relationship_patterns_validate(self):
        fixtures = [
            ([{"label": "Role", "name": "Engineer"}, {"label": "Department", "name": "Engineering"}],
             {"source_label": "Role", "source_name": "Engineer", "type": "BELONGS_TO", "target_label": "Department", "target_name": "Engineering"}),
            ([{"label": "Role", "name": "Engineer"}, {"label": "Skill", "name": "Python"}],
             {"source_label": "Role", "source_name": "Engineer", "type": "REQUIRES", "target_label": "Skill", "target_name": "Python"}),
            ([{"label": "Policy", "name": "Remote work"}, {"label": "Role", "name": "Engineer"}],
             {"source_label": "Policy", "source_name": "Remote work", "type": "APPLIES_TO", "target_label": "Role", "target_name": "Engineer"}),
            ([{"label": "Plan", "name": "Health Plus"}, {"label": "Service", "name": "Dental"}],
             {"source_label": "Plan", "source_name": "Health Plus", "type": "COVERS", "target_label": "Service", "target_name": "Dental"}),
            ([{"label": "Company", "name": "Zava"}, {"label": "Department", "name": "Engineering"}],
             {"source_label": "Company", "source_name": "Zava", "type": "HAS_DEPARTMENT", "target_label": "Department", "target_name": "Engineering"}),
            ([{"label": "Company", "name": "Zava"}, {"label": "Plan", "name": "Health Plus"}],
             {"source_label": "Company", "source_name": "Zava", "type": "OFFERS", "target_label": "Plan", "target_name": "Health Plus"}),
            ([{"label": "Policy", "name": "Remote work"}, {"label": "ApplicabilityGroup", "name": "Managers"}],
             {"source_label": "Policy", "source_name": "Remote work", "type": "APPLIES_TO", "target_label": "ApplicabilityGroup", "target_name": "Managers"}),
            ([{"label": "Employee", "name": "Alex"}, {"label": "ApplicabilityGroup", "name": "Managers"}],
             {"source_label": "Employee", "source_name": "Alex", "type": "APPLIES_ON", "target_label": "ApplicabilityGroup", "target_name": "Managers"}),
            ([{"label": "Employee", "name": "Alex"}, {"label": "Role", "name": "Engineer"}],
             {"source_label": "Employee", "source_name": "Alex", "type": "HAS_ROLE", "target_label": "Role", "target_name": "Engineer"}),
            ([{"label": "Plan", "name": "Health Plus"}, {"label": "Service", "name": "Dental"}],
             {"source_label": "Plan", "source_name": "Health Plus", "type": "EXCLUDES", "target_label": "Service", "target_name": "Dental"}),
        ]
        for entities, relationship in fixtures:
            with self.subTest(relationship=relationship["type"]):
                self.assertEqual(validate_extraction({"entities": entities, "relationships": [relationship]}),
                                 {"entities": entities, "relationships": [relationship]})

    def test_rejects_unapproved_labels_relationships_and_missing_endpoints(self):
        for extraction in (
            {"entities": [{"label": "UnknownEntity", "name": "A"}], "relationships": []},
            {"entities": [{"label": "Role", "name": "Engineer"}], "relationships": [{
                "source_label": "Role", "source_name": "Engineer", "type": "DELETE",
                "target_label": "Skill", "target_name": "Python",
            }]},
            {"entities": [{"label": "Role", "name": "Engineer"}], "relationships": [{
                "source_label": "Role", "source_name": "Engineer", "type": "REQUIRES",
                "target_label": "Skill", "target_name": "Python",
            }]},
            {"entities": [{"label": [], "name": "Bad"}], "relationships": []},
        ):
            with self.subTest(extraction=extraction), self.assertRaises(ValueError):
                validate_extraction(extraction)

    def test_extracts_and_caches_one_chunk(self):
        with TemporaryDirectory() as directory:
            models = Mock()
            models.chat.return_value = json.dumps({
                "entities": [{"label": "Role", "name": "Engineer"}, {"label": "Skill", "name": "Python"}],
                "relationships": [{"source_label": "Role", "source_name": "Engineer", "type": "REQUIRES", "target_label": "Skill", "target_name": "Python"}],
            })
            chunk = {"chunk_id": "chunk-1", "source_document": "roles.pdf", "pages": [1],
                     "text": "Engineer requires Python.", "figures": [{"caption": "Diagram"}]}
            first = extract_chunk(chunk, models, Path(directory), CONFIGURATION)
            second = extract_chunk(chunk, models, Path(directory), CONFIGURATION)
            self.assertEqual(first, second)
            self.assertEqual(first["chunk_id"], "chunk-1")
            models.chat.assert_called_once()
            self.assertIn("Diagram", models.chat.call_args.args[0])

    def test_invalid_response_is_not_cached(self):
        with TemporaryDirectory() as directory:
            models = Mock()
            models.chat.return_value = '{"entities":[{"label":"UnknownEntity","name":"Alex"}],"relationships":[]}'
            chunk = {"chunk_id": "chunk-1", "source_document": "roles.pdf", "text": "Source.", "pages": []}
            with self.assertRaises(ValueError):
                extract_chunk(chunk, models, Path(directory), CONFIGURATION)
            self.assertEqual(list(Path(directory).glob("*.json")), [])

    def test_duplicate_merging_is_case_and_whitespace_only(self):
        records = [
            record("one", [{"label": "Skill", "name": "Python"}, {"label": "Role", "name": " Data Engineer "}], [
                {"source_label": "Role", "source_name": " Data Engineer ", "type": "REQUIRES", "target_label": "Skill", "target_name": "Python"},
            ]),
            record("two", [{"label": "Skill", "name": " python "}, {"label": "Role", "name": "data   engineer"}], [
                {"source_label": "Role", "source_name": "data   engineer", "type": "REQUIRES", "target_label": "Skill", "target_name": " python "},
            ]),
            record("three", [{"label": "Service", "name": "Python"}], []),
        ]
        merged = merge_duplicate_entities(records)
        self.assertEqual(merged[0]["entities"], merged[1]["entities"])
        self.assertEqual(merged[0]["entities"], [
            {"label": "Role", "name": "Data Engineer"}, {"label": "Skill", "name": "Python"},
        ])
        self.assertEqual(merged[0]["relationships"], merged[1]["relationships"])
        self.assertEqual(merged[2]["entities"], [{"label": "Service", "name": "Python"}])

    def test_merging_rejects_records_outside_the_schema(self):
        with self.assertRaises(ValueError):
            merge_duplicate_entities([record("bad", [{"label": "UnknownEntity", "name": "A"}], [])])

    def test_unknown_relationships_remain_rejected(self):
        with self.assertRaises(ValueError):
            validate_extraction({
                "entities": [{"label": "Plan", "name": "Health Plus"},
                             {"label": "Service", "name": "Dental"}],
                "relationships": [{
                    "source_label": "Plan", "source_name": "Health Plus",
                    "type": "GRANTS", "target_label": "Service", "target_name": "Dental",
                }],
            })


if __name__ == "__main__":
    unittest.main()