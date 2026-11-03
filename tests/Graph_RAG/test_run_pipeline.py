
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphrag.run_pipeline import main


class TestRunPipeline(unittest.TestCase):

    @patch("graphrag.run_pipeline.CompositeDocumentProcessingPipeline")
    @patch("graphrag.run_pipeline.DatabaseMcpClient")
    @patch("graphrag.run_pipeline.DefaultAzureCredential")
    @patch("sys.argv", ["run_pipeline.py", "--blob", "role_library.pdf"])
    def test_main_connects_components(
        self,
        mock_credential_class,
        mock_client_factory,
        mock_pipeline_class,
    ):
        client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = client

        main()

        client.health_check.assert_called_once()
        client.ensure_schema.assert_called_once()
        self.assertIs(
            mock_pipeline_class.call_args.kwargs["chunk_loader"].client, client,
        )
        mock_pipeline_class.return_value.load_document.assert_called_once_with(
            "role_library.pdf"
        )
        mock_credential_class.return_value.close.assert_called_once()

    @patch("graphrag.run_pipeline.CompositeDocumentProcessingPipeline")
    @patch("graphrag.run_pipeline.DatabaseMcpClient")
    @patch("graphrag.run_pipeline.DefaultAzureCredential")
    @patch("sys.argv", ["run_pipeline.py", "--blob", "role_library.pdf"])
    def test_mcp_failure_closes_azure_credential(
        self, credential, client_factory, pipeline,
    ):
        client_factory.return_value.__enter__.return_value.health_check.side_effect = (
            RuntimeError("MCP unavailable")
        )
        with self.assertRaisesRegex(RuntimeError, "MCP unavailable"):
            main()
        pipeline.assert_not_called()
        credential.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
