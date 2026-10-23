
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from graphrag.config import settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pipeline import main


class TestRunPipeline(unittest.TestCase):

    @patch("scripts.run_pipeline.CompositeDocumentProcessingPipeline")
    @patch("scripts.run_pipeline.create_neo4j_driver")
    @patch("scripts.run_pipeline.create_neo4j_schema")
    @patch("scripts.run_pipeline.DefaultAzureCredential")
    @patch("sys.argv", ["run_pipeline.py", "--blob", "role_library.pdf"])
    def test_main_connects_components(
        self,
        mock_credential_class,
        mock_schema,
        mock_driver_factory,
        mock_pipeline_class,
    ):
        driver = MagicMock()
        session = MagicMock()

        mock_driver_factory.return_value.__enter__.return_value = driver
        driver.session.return_value.__enter__.return_value = session

        main()

        driver.verify_connectivity.assert_called_once()
        driver.session.assert_called_once_with(database=settings.neo4j.database)
        mock_schema.assert_called_once_with(session)
        mock_pipeline_class.return_value.load_document.assert_called_once_with(
            "role_library.pdf"
        )
        mock_credential_class.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
