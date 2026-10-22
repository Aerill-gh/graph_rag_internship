
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from graphrag.components import (
    ZavaChunkEmbedder,
    Neo4jChunkLoader,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag.components import ZavaDocumentParser, ZavaDocumentSplitter
from graphrag.pipeline import ParsedDocument
from graphrag.pipeline import Chunk


class TestZavaDocumentParser(unittest.TestCase):

    def test_pdf_parsing(self):
        pdf_parser = Mock(return_value={
            "pages": [{"page": 1, "text": "Example PDF"}],
            "figures": [],
        })

        parser = ZavaDocumentParser(pdf_parser=pdf_parser)
        result = parser.parse_document(b"PDF bytes", "role_library.pdf")

        pdf_parser.assert_called_once_with(b"PDF bytes")
        self.assertEqual(result.document_type, "pdf")
        self.assertEqual(result.source_document, "role_library.pdf")
        self.assertEqual(result.pages[0]["text"], "Example PDF")

    def test_markdown_parsing(self):
        parser = ZavaDocumentParser(pdf_parser=Mock())
        result = parser.parse_document(
            b"## Company Overview\nExample content",
            "Zava_Company_Overview.md",
        )

        self.assertEqual(result.document_type, "markdown")
        self.assertEqual(result.source_document, "Zava_Company_Overview.md")
        self.assertIn("Company Overview", result.text)

    def test_unsupported_document(self):
        parser = ZavaDocumentParser(pdf_parser=Mock())

        with self.assertRaises(ValueError):
            parser.parse_document(b"content", "example.txt")


class TestZavaDocumentSplitter(unittest.TestCase):

    def test_pdf_splitting(self):
        pdf_splitter = Mock(return_value=[
            {"chunk_id": "role_library_001", "text": "Example PDF"}
        ])
        markdown_splitter = Mock()

        splitter = ZavaDocumentSplitter(pdf_splitter, markdown_splitter)

        parsed = ParsedDocument(
            source_document="role_library.pdf",
            document_type="pdf",
            pages=[{"page": 1, "text": "Example PDF"}],
        )

        result = splitter.split_document(parsed)

        pdf_splitter.assert_called_once_with(
            parsed.pages,
            parsed.figures,
            "role_library.pdf",
        )
        markdown_splitter.assert_not_called()
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Chunk)
        self.assertEqual(result[0].text, "Example PDF")

    def test_markdown_splitting(self):
        pdf_splitter = Mock()
        markdown_splitter = Mock(return_value=[
            {"chunk_id": "overview_001", "text": "Example Markdown"}
        ])

        splitter = ZavaDocumentSplitter(pdf_splitter, markdown_splitter)

        parsed = ParsedDocument(
            source_document="Zava_Company_Overview.md",
            document_type="markdown",
            text="## Overview\nExample Markdown",
        )

        result = splitter.split_document(parsed)

        markdown_splitter.assert_called_once_with(
            parsed.text.encode("utf-8"),
            parsed.source_document,
        )
        pdf_splitter.assert_not_called()
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Chunk)
        self.assertEqual(result[0].text, "Example Markdown")

    def test_unsupported_document_type(self):
        splitter = ZavaDocumentSplitter(Mock(), Mock())

        parsed = ParsedDocument(
            source_document="example.txt",
            document_type="txt",
        )

        with self.assertRaises(ValueError):
            splitter.split_document(parsed)



class TestZavaChunkEmbedder(unittest.TestCase):

    @patch("graphrag.components.get_chunk_embedding")
    def test_embed_chunk(self, mock_embedding):
        chunk = Chunk(chunk_id="test_001", text="Example chunk")
        expected_embedding = [0.1, 0.2]
        mock_embedding.return_value = expected_embedding

        embedder = ZavaChunkEmbedder()
        result = embedder.embed_chunk(chunk)

        mock_embedding.assert_called_once_with(chunk.to_record())
        self.assertEqual(result, expected_embedding)


class TestNeo4jChunkLoader(unittest.TestCase):

    @patch("graphrag.components.store_chunk")
    def test_load_chunk(self, mock_store):
        session = Mock()
        chunk = Chunk(chunk_id="test_001", text="Example chunk")
        embedding = [0.1, 0.2]
        chunk.set_embedding(embedding)

        loader = Neo4jChunkLoader(session)
        loader.load_chunk(chunk)

        mock_store.assert_called_once_with(
            session,
            chunk.to_record(),
            embedding,
        )



if __name__ == "__main__":
    unittest.main()
