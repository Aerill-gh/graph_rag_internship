import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call
from graphrag.components import (
    ZavaDocumentParser,
    ZavaDocumentSplitter,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.pipeline import CompositeDocumentProcessingPipeline


class TestCompositeDocumentProcessingPipeline(unittest.TestCase):

    def setUp(self):
        self.loader = Mock()
        self.parser = Mock()
        self.splitter = Mock()
        self.embedder = Mock()
        self.chunk_loader = Mock()

        self.pipeline = CompositeDocumentProcessingPipeline(
            document_loader=self.loader,
            document_parser=self.parser,
            document_splitter=self.splitter,
            chunk_embedder=self.embedder,
            chunk_loader=self.chunk_loader,
        )

    def test_pipeline_processes_document(self):
        self.loader.load.return_value = b"Example document"

        parsed_document = Mock()
        self.parser.parse_document.return_value = parsed_document

        chunks = [
            {"chunk_id": "chunk_001", "text": "First chunk"},
            {"chunk_id": "chunk_002", "text": "Second chunk"},
        ]
        self.splitter.split_document.return_value = chunks

        self.embedder.embed_chunk.side_effect = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        self.pipeline.load_document("example.pdf")

        self.loader.load.assert_called_once_with("example.pdf")
        self.parser.parse_document.assert_called_once_with(
            b"Example document",
            "example.pdf",
        )
        self.splitter.split_document.assert_called_once_with(
            parsed_document
        )

        self.assertEqual(
            self.embedder.embed_chunk.call_args_list,
            [call(chunks[0]), call(chunks[1])],
        )

        self.assertEqual(
            self.chunk_loader.load_chunk.call_args_list,
            [
                call(chunks[0], [0.1, 0.2]),
                call(chunks[1], [0.3, 0.4]),
            ],
        )

    def test_pipeline_handles_empty_chunks(self):
        self.loader.load.return_value = b"Empty document"
        self.parser.parse_document.return_value = Mock()
        self.splitter.split_document.return_value = []

        self.pipeline.load_document("empty.pdf")

        self.embedder.embed_chunk.assert_not_called()
        self.chunk_loader.load_chunk.assert_not_called()


    def test_pipeline_with_zava_adapters(self):
        pdf_parser = Mock(return_value={
            "pages": [{"page": 1, "text": "Example PDF"}],
            "figures": [],
        })

        pdf_splitter = Mock(return_value=[
            {
                "chunk_id": "example_001",
                "source_document": "example.pdf",
                "pages": [1],
                "text": "Example PDF",
                "token_count": 2,
                "figures": [],
            }
        ])

        markdown_splitter = Mock()

        parser = ZavaDocumentParser(pdf_parser)
        splitter = ZavaDocumentSplitter(
            pdf_splitter,
            markdown_splitter,
        )

        self.loader.load.return_value = b"PDF bytes"
        self.embedder.embed_chunk.return_value = [0.1, 0.2]

        pipeline = CompositeDocumentProcessingPipeline(
            document_loader=self.loader,
            document_parser=parser,
            document_splitter=splitter,
            chunk_embedder=self.embedder,
            chunk_loader=self.chunk_loader,
        )

        pipeline.load_document("example.pdf")

        pdf_parser.assert_called_once_with(b"PDF bytes")
        pdf_splitter.assert_called_once()

        self.embedder.embed_chunk.assert_called_once()
        self.chunk_loader.load_chunk.assert_called_once()


if __name__ == "__main__":
    unittest.main()
