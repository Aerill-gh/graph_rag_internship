import asyncio
import socket
import unittest
from unittest.mock import Mock, patch

import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session

from Chatbot.document_tools import DocumentPassage, McpDocumentSearch
from Chatbot.mcp_server import create_mcp_server, retrieve_documents


PASSAGE = DocumentPassage(
    chunk_id="test-1",
    document="example.pdf",
    pages=[4],
    text="Offline example evidence, not a real company policy.",
)


class TestMcpServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.search = Mock(return_value=[PASSAGE])
        self.server = create_mcp_server(self.search)

    async def test_discovery_and_client_search(self):
        async with create_connected_server_and_client_session(self.server) as session:
            tools = (await session.list_tools()).tools
            self.assertEqual([tool.name for tool in tools], ["search_documents"])
            schema = tools[0].inputSchema
            self.assertEqual(schema["required"], ["question"])
            self.assertEqual(schema["properties"]["question"]["type"], "string")
            self.assertIn("passages", tools[0].outputSchema["properties"])
            self.assertTrue(tools[0].annotations.readOnlyHint)
            results = await McpDocumentSearch(session).search_documents("Example question")
            self.assertEqual(results, [PASSAGE])
            self.search.assert_called_once_with("Example question")

    async def test_empty_results(self):
        self.search.return_value = []
        async with create_connected_server_and_client_session(self.server) as session:
            self.assertEqual(
                await McpDocumentSearch(session).search_documents("No matches"), [],
            )

    async def test_invalid_arguments_are_tool_errors(self):
        async with create_connected_server_and_client_session(self.server) as session:
            for arguments in ({}, {"question": 9}, {"question": " \n "}):
                with self.subTest(arguments=arguments):
                    result = await session.call_tool("search_documents", arguments)
                    self.assertTrue(result.isError)
        self.search.assert_not_called()

    async def test_retrieval_failure_is_not_empty_evidence(self):
        self.search.side_effect = RuntimeError("Retrieval unavailable")
        async with create_connected_server_and_client_session(self.server) as session:
            with self.assertRaisesRegex(RuntimeError, "MCP document-search tool"):
                await McpDocumentSearch(session).search_documents("Example question")

    async def test_malformed_retrieval_output_is_an_error(self):
        self.search.return_value = [{"document": "missing-fields.pdf"}]
        async with create_connected_server_and_client_session(self.server) as session:
            with self.assertRaisesRegex(RuntimeError, "MCP document-search tool"):
                await McpDocumentSearch(session).search_documents("Example question")

    async def test_streamable_http_client_round_trip(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        self.addCleanup(sock.close)
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(
            self.server.streamable_http_app(), log_level="warning",
        ))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                        self.fail("MCP server exited before startup.")
                    await asyncio.sleep(0.01)
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.assertEqual(
                        await McpDocumentSearch(session).search_documents("HTTP question"),
                        [PASSAGE],
                    )
            self.search.assert_called_once_with("HTTP question")
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=10)


class TestRetrievalAdapter(unittest.TestCase):
    def test_delegates_and_normalizes_search_results(self):
        retrieval = Mock()
        retrieval.search_documents.return_value = [
            {**PASSAGE.model_dump(), "score": 0.9},
        ]
        with patch.dict("sys.modules", {"graphrag.tools": retrieval}):
            self.assertEqual(retrieve_documents("Example question"), [PASSAGE])
        retrieval.search_documents.assert_called_once_with("Example question")


if __name__ == "__main__":
    unittest.main()
