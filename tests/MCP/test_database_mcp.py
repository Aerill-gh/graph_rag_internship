import ast
import asyncio
import socket
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import uvicorn
from anyio import to_thread
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from database_mcp.client import DatabaseMcpClient
from database_mcp.config import load_database_settings
from database_mcp.contracts import (
    EMBEDDING_DIMENSIONS,
    ChunkPayload,
    ChunkReadResponse,
    ChunkWrite,
    ChunkWriteResponse,
    ExtractionWrite,
    ExtractionWriteResponse,
    HealthResponse,
    SchemaResponse,
)
from database_mcp.server import DatabaseBackend, create_mcp_server


CHUNK = {
    "chunk_id": "test-one", "source_document": "example.pdf",
    "text": "Offline test evidence.", "pages": [1], "token_count": 3, "figures": [],
}
EXTRACTION = {
    "entities": [
        {"label": "Role", "name": "Engineer"}, {"label": "Skill", "name": "Python"},
    ],
    "relationships": [{
        "source_label": "Role", "source_name": "Engineer", "type": "REQUIRES",
        "target_label": "Skill", "target_name": "Python",
    }],
}


def backend_mock() -> Mock:
    backend = Mock(spec=DatabaseBackend)
    backend.health_check.return_value = HealthResponse()
    backend.ensure_schema.return_value = SchemaResponse()
    backend.upsert_chunk.return_value = ChunkWriteResponse(chunk_id=CHUNK["chunk_id"])
    backend.upsert_extraction.return_value = ExtractionWriteResponse(
        chunk_id=CHUNK["chunk_id"], entity_mentions=2, relationship_mentions=1,
    )
    backend.get_chunk.return_value = ChunkReadResponse(
        chunk=ChunkPayload.model_validate(CHUNK),
        embedding=[0.1] * EMBEDDING_DIMENSIONS, embedding_model="test-embedding",
        entities=EXTRACTION["entities"], relationships=EXTRACTION["relationships"],
        entities_truncated=False, relationships_truncated=False,
    )
    return backend


class ContractTests(unittest.TestCase):
    def test_chunk_validation_rejects_bad_fields_and_vectors(self):
        for chunk in (
            {**CHUNK, "chunk_id": " "}, {**CHUNK, "pages": [True]},
            {**CHUNK, "pages": [0]}, {**CHUNK, "token_count": -1},
            {**CHUNK, "unknown": "field"},
        ):
            with self.subTest(chunk=chunk), self.assertRaises(ValidationError):
                ChunkWrite.model_validate({"chunk": chunk})
        for embedding in ([0.1], [float("nan")] * EMBEDDING_DIMENSIONS):
            with self.subTest(embedding=embedding[:1]), self.assertRaises(ValidationError):
                ChunkWrite.model_validate({
                    "chunk": CHUNK, "embedding": embedding, "embedding_model": "test",
                })
        with self.assertRaises(ValidationError):
            ChunkWrite.model_validate({
                "chunk": CHUNK, "embedding": [0.1] * EMBEDDING_DIMENSIONS,
            })

    def test_extraction_rejects_unknown_labels_and_missing_endpoints(self):
        for extraction in (
            {"entities": [{"label": "Unknown", "name": "X"}], "relationships": []},
            {**EXTRACTION, "entities": EXTRACTION["entities"][:1]},
        ):
            with self.subTest(extraction=extraction), self.assertRaises(ValidationError):
                ExtractionWrite.model_validate({
                    "chunk_id": CHUNK["chunk_id"], "extraction": extraction,
                })

    def test_database_configuration_does_not_fall_back_to_shared_credentials(self):
        with patch("database_mcp.config.dotenv_values", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "env.database-mcp"):
                load_database_settings()


class McpServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = backend_mock()
        self.server = create_mcp_server(self.backend)

    async def test_discovery_and_read_write_calls(self):
        async with create_connected_server_and_client_session(self.server) as session:
            tools = {tool.name: tool for tool in (await session.list_tools()).tools}
            self.assertEqual(set(tools), {
                "health_check", "ensure_schema", "upsert_chunk",
                "upsert_extraction", "get_chunk",
            })
            self.assertTrue(tools["get_chunk"].annotations.readOnlyHint)
            self.assertFalse(tools["upsert_extraction"].annotations.readOnlyHint)
            chunk_request = ChunkWrite.model_validate({"chunk": CHUNK})
            extraction_request = ExtractionWrite.model_validate({
                "chunk_id": CHUNK["chunk_id"], "extraction": EXTRACTION,
            })
            for name, arguments in (
                ("health_check", {}), ("ensure_schema", {}),
                ("upsert_chunk", {"request": chunk_request.model_dump()}),
                ("upsert_extraction", {"request": extraction_request.model_dump()}),
                ("get_chunk", {"chunk_id": CHUNK["chunk_id"]}),
            ):
                result = await session.call_tool(name, arguments)
                self.assertFalse(result.isError, name)
            self.backend.upsert_chunk.assert_called_once_with(chunk_request)
            self.backend.upsert_extraction.assert_called_once_with(extraction_request)

    async def test_server_rejects_invalid_write_before_backend_call(self):
        async with create_connected_server_and_client_session(self.server) as session:
            for name, arguments in (
                ("upsert_chunk", {"request": {"chunk": {**CHUNK, "text": " "}}}),
                ("upsert_chunk", {"request": {
                    "chunk": CHUNK, "embedding": [0.1], "embedding_model": "test",
                }}),
                ("upsert_extraction", {"request": {
                    "chunk_id": CHUNK["chunk_id"],
                    "extraction": {"entities": [], "relationships": EXTRACTION["relationships"]},
                }}),
            ):
                result = await session.call_tool(name, arguments)
                self.assertTrue(result.isError, name)
        self.backend.upsert_chunk.assert_not_called()
        self.backend.upsert_extraction.assert_not_called()

    async def test_missing_source_and_backend_failures_are_errors(self):
        self.backend.upsert_extraction.side_effect = ValueError("Source chunk missing")
        self.backend.health_check.side_effect = RuntimeError("Database unavailable")
        async with create_connected_server_and_client_session(self.server) as session:
            for name, arguments in (
                ("health_check", {}),
                ("upsert_extraction", {"request": {
                    "chunk_id": CHUNK["chunk_id"], "extraction": EXTRACTION,
                }}),
            ):
                self.assertTrue((await session.call_tool(name, arguments)).isError)

    async def test_http_sync_client_round_trip_and_failure(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        self.addCleanup(sock.close)
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(
            self.server.streamable_http_app(), log_level="warning",
        ))
        task = asyncio.create_task(server.serve(sockets=[sock]))

        def round_trip():
            with DatabaseMcpClient(f"http://127.0.0.1:{port}/mcp") as client:
                client.health_check()
                client.ensure_schema()
                for _ in range(2):
                    client.store_chunk(
                        CHUNK, [0.1] * EMBEDDING_DIMENSIONS,
                        embedding_model="test-embedding",
                    )
                    client.store_extraction(CHUNK["chunk_id"], EXTRACTION)
                self.assertEqual(client.get_chunk(CHUNK["chunk_id"]).chunk.text, CHUNK["text"])
                self.backend.upsert_chunk.side_effect = RuntimeError("Write unavailable")
                with self.assertRaisesRegex(RuntimeError, "upsert_chunk"):
                    client.store_chunk(CHUNK)

        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                        self.fail("MCP server exited before startup.")
                    await asyncio.sleep(0.01)
            async with asyncio.timeout(20):
                await to_thread.run_sync(round_trip)
            self.assertEqual(self.backend.upsert_extraction.call_count, 2)
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=10)


class ClientResponseTests(unittest.TestCase):
    def client(self, result: CallToolResult) -> DatabaseMcpClient:
        client = DatabaseMcpClient("http://127.0.0.1:8002/mcp")
        client._portal = Mock()
        client._session = Mock()
        client._portal.call.return_value = result
        return client

    def test_client_rejects_errors_and_malformed_success(self):
        results = [
            CallToolResult(isError=True, content=[TextContent(type="text", text="error")]),
            CallToolResult(content=[TextContent(type="text", text='{"status":"failed"}')]),
            CallToolResult(content=[]),
        ]
        for result in results:
            with self.subTest(result=result), self.assertRaises((RuntimeError, ValueError)):
                self.client(result).health_check()

    def test_client_rejects_wrong_chunk_acknowledgement(self):
        result = CallToolResult(
            content=[], structuredContent={"chunk_id": "different"},
        )
        with self.assertRaisesRegex(ValueError, "different source chunk"):
            self.client(result).store_chunk(CHUNK)


class IngestionBoundaryTests(unittest.TestCase):
    def test_ingestion_has_no_driver_imports_or_credential_access(self):
        root = Path(__file__).resolve().parents[2]
        files = (
            root / "src" / "graphrag" / "build_graph.py",
            root / "src" / "graphrag" / "run_pipeline.py",
            root / "src" / "graphrag" / "components.py",
            root / "src" / "graphrag" / "check_config.py",
            root / "src" / "Neo4j" / "ingest_neo4j.py",
            root / "src" / "database_mcp" / "client.py",
        )
        for path in files:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotIn(node.module, {
                            "neo4j", "Neo4j.neo4j_store", "database_mcp.neo4j_backend",
                        })
                    elif isinstance(node, ast.Import):
                        self.assertFalse(any(
                            alias.name == "neo4j" or alias.name.startswith("Neo4j.")
                            for alias in node.names
                        ))
                for forbidden in (
                    "settings.neo4j", "create_neo4j_driver", "GraphDatabase",
                    "NEO4J_PASSWORD", "NEO4J_URI", "NEO4J_USERNAME",
                ):
                    self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
