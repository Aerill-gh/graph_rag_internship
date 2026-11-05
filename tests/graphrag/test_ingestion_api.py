import tempfile
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from fastapi.testclient import TestClient

from graphrag import ingestion_api
from graphrag.build_graph import build_graph_for_chunks
from graphrag.database_mcp.client import DatabaseMcpError


class IngestionApiTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.enterContext(patch.object(ingestion_api, "DOCUMENT_DIR", self.root))
        self.lock = Lock()
        self.enterContext(patch.object(ingestion_api, "ingestion_lock", self.lock))
        self.credential_factory = self.enterContext(
            patch.object(ingestion_api, "DefaultAzureCredential")
        )
        self.database_factory = self.enterContext(
            patch.object(ingestion_api, "DatabaseMcpClient")
        )
        self.database = self.database_factory.return_value.__enter__.return_value
        self.build = self.enterContext(
            patch.object(ingestion_api, "build_graph_for_chunks", return_value={
                "chunks": 1, "entity_mentions": 2, "relationship_mentions": 1,
            })
        )
        self.pdf_parser = self.enterContext(
            patch.object(ingestion_api, "parse_pdf_document", return_value={
                "pages": [{"page": 1, "text": "Engineer requires Python."}],
                "figures": [],
            })
        )
        self.client = self.enterContext(TestClient(ingestion_api.app))
        self.markdown_name = "Zava_Company_Overview.md"
        (self.root / self.markdown_name).write_text(
            "# Zava\nEngineer requires Python.", encoding="utf-8",
        )

    def test_database_backend_imports_real_storage_helpers(self):
        from graphrag.database_mcp import neo4j_backend
        from graphrag.Neo4j import neo4j_store

        self.assertIs(neo4j_backend.create_neo4j_schema, neo4j_store.create_neo4j_schema)
        self.assertIs(neo4j_backend.store_chunk, neo4j_store.store_chunk)
        self.assertIs(neo4j_backend.store_extraction, neo4j_store.store_extraction)

    def ingest(self, filename=None):
        return self.client.post(
            "/documents/ingest",
            json={"filename": filename or self.markdown_name},
        )

    def test_list_reports_all_seven_documents_and_availability(self):
        response = self.client.get("/documents")
        self.assertEqual(response.status_code, 200, response.text)
        documents = response.json()
        self.assertEqual(len(documents), 7)
        self.assertEqual(
            [item["filename"] for item in documents if item["available"]],
            [self.markdown_name],
        )
        self.database_factory.assert_not_called()
        self.credential_factory.assert_not_called()

    def test_markdown_preserves_source_and_returns_build_counts(self):
        response = self.ingest()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {
            "status": "completed", "document": self.markdown_name,
            "chunks": 1, "entity_mentions": 2, "relationship_mentions": 1,
        })
        self.database.health_check.assert_called_once_with()
        self.database.ensure_schema.assert_called_once_with()
        client, chunks = self.build.call_args.args
        self.assertIs(client, self.database)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source_document"], self.markdown_name)
        self.assertEqual(chunks[0]["chunk_id"], "Zava_Company_Overview_001")
        self.assertEqual(chunks[0]["pages"], [])
        self.assertIn("Engineer requires Python.", chunks[0]["text"])
        self.pdf_parser.assert_not_called()
        self.database_factory.return_value.__exit__.assert_called_once()
        self.credential_factory.return_value.__exit__.assert_called_once()
        self.assertFalse(self.lock.locked())

    def test_pdf_uses_existing_parser_and_preserves_page(self):
        (self.root / "role_library.pdf").write_bytes(b"fake PDF for mocked parser")
        response = self.ingest("role_library.pdf")
        self.assertEqual(response.status_code, 200, response.text)
        self.pdf_parser.assert_called_once_with(
            b"fake PDF for mocked parser",
            credential=self.credential_factory.return_value.__enter__.return_value,
        )
        chunk = self.build.call_args.args[1][0]
        self.assertEqual(chunk["source_document"], "role_library.pdf")
        self.assertEqual(chunk["pages"], [1])

    def test_existing_graph_builder_writes_vectors_and_source_backed_facts(self):
        def build(client, chunks):
            def extract(chunk):
                return {
                    "chunk_id": chunk["chunk_id"],
                    "entities": [
                        {"label": "Role", "name": "Engineer"},
                        {"label": "Skill", "name": "Python"},
                    ],
                    "relationships": [{
                        "source_label": "Role", "source_name": "Engineer",
                        "type": "REQUIRES", "target_label": "Skill",
                        "target_name": "Python",
                    }],
                }

            return build_graph_for_chunks(
                client, chunks, embedding_fn=lambda _chunk: [0.1] * 1536,
                extraction_fn=extract,
            )

        self.build.side_effect = build
        with patch("graphrag.build_graph.settings") as settings:
            settings.azure_openai_embedding_deployment = "test-embedding"
            response = self.ingest()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["relationship_mentions"], 1)
        stored_chunk = self.database.store_chunk.call_args.args[0]
        self.assertEqual(stored_chunk["source_document"], self.markdown_name)
        self.assertEqual(len(self.database.store_chunk.call_args.args[1]), 1536)
        self.assertEqual(
            self.database.store_chunk.call_args.kwargs,
            {"embedding_model": "test-embedding"},
        )
        chunk_id, extraction = self.database.store_extraction.call_args.args
        self.assertEqual(chunk_id, stored_chunk["chunk_id"])
        self.assertEqual(extraction["relationships"][0]["type"], "REQUIRES")

    def test_invalid_filenames_and_extra_fields_are_rejected(self):
        for body in (
            {}, {"filename": ""}, {"filename": "../role_library.pdf"},
            {"filename": r"..\role_library.pdf"},
            {"filename": r"C:\private\role_library.pdf"},
            {"filename": "other.pdf"}, {"filename": "document.exe"},
            {"filename": self.markdown_name, "path": "other"},
        ):
            with self.subTest(body=body):
                response = self.client.post("/documents/ingest", json=body)
                self.assertEqual(response.status_code, 422, response.text)
        self.build.assert_not_called()
        self.database_factory.assert_not_called()

    def test_missing_document_is_404_and_releases_lock(self):
        with self.assertLogs("graphrag.ingestion_api", level="ERROR"):
            response = self.ingest("role_library.pdf")
        self.assertEqual(response.status_code, 404, response.text)
        self.database_factory.assert_not_called()
        self.assertEqual(self.ingest().status_code, 200)

    def test_empty_document_is_rejected_before_services_are_opened(self):
        (self.root / self.markdown_name).write_bytes(b" \r\n")
        response = self.ingest()
        self.assertEqual(response.status_code, 422, response.text)
        self.database_factory.assert_not_called()
        self.build.assert_not_called()
        self.assertFalse(self.lock.locked())

    def test_unreadable_markdown_is_422(self):
        (self.root / self.markdown_name).write_bytes(b"\xff")
        with self.assertLogs("graphrag.ingestion_api", level="ERROR"):
            response = self.ingest()
        self.assertEqual(response.status_code, 422, response.text)
        self.build.assert_not_called()
        self.assertFalse(self.lock.locked())

    def test_no_chunks_is_not_reported_as_success(self):
        with patch.object(ingestion_api.ZavaDocumentSplitter, "split_document", return_value=[]):
            response = self.ingest()
        self.assertEqual(response.status_code, 422, response.text)
        self.build.assert_not_called()

    def test_concurrent_ingestion_is_rejected(self):
        self.lock.acquire()
        try:
            response = self.ingest()
            self.assertEqual(response.status_code, 409, response.text)
            self.database_factory.assert_not_called()
            self.assertTrue(self.lock.locked())
        finally:
            self.lock.release()

    def test_escaped_resolved_path_is_rejected(self):
        with patch.object(Path, "resolve", side_effect=[self.root, self.root.parent / "outside.pdf"]):
            response = self.ingest()
        self.assertEqual(response.status_code, 400, response.text)
        self.database_factory.assert_not_called()
        self.assertFalse(self.lock.locked())

    def test_mcp_failure_is_503_and_does_not_expose_details(self):
        self.database.health_check.side_effect = DatabaseMcpError("private connection detail")
        with self.assertLogs("graphrag.ingestion_api", level="ERROR") as logs:
            response = self.ingest()
        self.assertEqual(response.status_code, 503, response.text)
        self.assertNotIn("private connection detail", response.text)
        self.assertNotIn("private connection detail", "".join(logs.output))
        self.build.assert_not_called()
        self.assertFalse(self.lock.locked())
        self.database_factory.return_value.__exit__.assert_called_once()
        self.credential_factory.return_value.__exit__.assert_called_once()

    def test_build_failure_warns_about_partial_writes_and_allows_retry(self):
        self.build.side_effect = RuntimeError("private model detail")
        with self.assertLogs("graphrag.ingestion_api", level="ERROR") as logs:
            response = self.ingest()
        self.assertEqual(response.status_code, 502, response.text)
        self.assertIn("Earlier writes may remain", response.json()["detail"])
        self.assertNotIn("private model detail", response.text)
        self.assertNotIn("private model detail", "".join(logs.output))
        self.assertFalse(self.lock.locked())
        self.build.side_effect = None
        self.assertEqual(self.ingest().status_code, 200)

    def test_missing_cache_is_service_failure_not_missing_document(self):
        self.build.side_effect = FileNotFoundError("private cache path")
        with self.assertLogs("graphrag.ingestion_api", level="ERROR"):
            response = self.ingest()
        self.assertEqual(response.status_code, 503, response.text)
        self.assertNotIn("private cache path", response.text)
        self.assertFalse(self.lock.locked())


if __name__ == "__main__":
    unittest.main()
