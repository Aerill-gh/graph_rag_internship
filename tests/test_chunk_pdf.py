import json
import sys
import tempfile
import unittest
from io import BytesIO
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pypdf import PdfReader, PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

chunk_pdf = import_module("graphrag.documents")


class AnalyzeBatchTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.cache_dir = Path(temporary_directory.name)
        for name, value in (
            ("CACHE_DIR", self.cache_dir),
            ("FIGURE_DIR", self.cache_dir / "figures"),
        ):
            patcher = patch.object(chunk_pdf, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(width=100, height=100)
        buffer = BytesIO()
        writer.write(buffer)
        buffer.seek(0)
        self.reader = PdfReader(buffer)

        self.client = MagicMock()
        poller = self.client.begin_analyze_document.return_value
        poller.details = {"operation_id": "operation"}
        self.result = SimpleNamespace(
            pages=[SimpleNamespace(
                page_number=1, lines=[SimpleNamespace(content=" Page text ")]
            )],
            figures=[
                SimpleNamespace(
                    id="1.1",
                    bounding_regions=[SimpleNamespace(page_number=1)],
                    caption=SimpleNamespace(content="A diagram"),
                ),
                SimpleNamespace(id="1.2", bounding_regions=[], caption=None),
            ],
        )
        poller.result.return_value = self.result
        self.client.get_analyze_result_figure.return_value = [b"image", b"-bytes"]

    def analyze(self):
        return chunk_pdf.analyze_batch(self.client, self.reader, "pdf-hash", 1, 3)

    def test_analyzes_selected_pages_and_caches_figures(self):
        batch = self.analyze()
        self.assertEqual(batch["pages"], [{"page": 2, "text": "Page text"}])
        figure = batch["figures"][0]
        self.assertEqual(figure["page"], 2)
        self.assertEqual(figure["caption"], "A diagram")
        self.assertEqual(figure["id"], "1.1")
        self.assertEqual(Path(figure["image_path"]).read_bytes(), b"image-bytes")
        self.assertEqual(len(batch["figures"]), 1)
        body = self.client.begin_analyze_document.call_args.kwargs["body"]
        body.seek(0)
        self.assertEqual(len(PdfReader(body).pages), 2)
        cache_path = next(self.cache_dir.glob("*.json"))
        self.assertEqual(json.loads(cache_path.read_text()), batch)

    def test_reuses_cache_without_calling_azure(self):
        original = self.analyze()
        self.client.reset_mock()
        self.assertEqual(self.analyze(), original)
        self.client.begin_analyze_document.assert_not_called()
        self.client.get_analyze_result_figure.assert_not_called()

    def test_reanalyzes_when_cached_figure_is_missing(self):
        batch = self.analyze()
        Path(batch["figures"][0]["image_path"]).unlink()
        self.client.reset_mock()
        self.assertEqual(self.analyze(), batch)
        self.client.begin_analyze_document.assert_called_once()

    def test_missing_pages_raises_without_caching(self):
        self.result.pages = []
        with self.assertRaisesRegex(RuntimeError, "No pages returned for pages 2-3"):
            self.analyze()
        self.assertEqual(list(self.cache_dir.glob("*.json")), [])


class ChunkTests(unittest.TestCase):
    def test_pdf_chunks_preserve_overlap_and_sources(self):
        pages = [{"page": 1, "text": "alpha beta gamma delta epsilon zeta"}]
        with patch.object(chunk_pdf, "MAX_TOKENS", 4), patch.object(
            chunk_pdf, "OVERLAP_TOKENS", 1
        ):
            chunks = chunk_pdf.create_chunks(pages, [], "folder/example.pdf")
        tokens = chunk_pdf.encoder.encode(pages[0]["text"])
        self.assertEqual(len(chunks), 2)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk["chunk_id"], f"folder_example_{index + 1:03d}")
            self.assertEqual(chunk["source_document"], "folder/example.pdf")
            self.assertEqual(chunk["pages"], [1])
            self.assertEqual(chunk["text"], chunk_pdf.encoder.decode(tokens[index * 3:index * 3 + 4]))

    def test_figure_caption_is_used_for_an_empty_page(self):
        figure = {"id": "1.1", "page": 2, "caption": "Diagram", "image_path": "figure.png"}
        chunks = chunk_pdf.create_chunks([{"page": 2, "text": ""}], [figure], "test.pdf")
        self.assertEqual(chunks[0]["text"], "Diagram")
        self.assertEqual(chunks[0]["figures"], [figure])

    def test_markdown_preserves_section_boundaries(self):
        chunks = chunk_pdf.create_markdown_chunks(
            b"\xef\xbb\xbf# Title\n\n## First\nHello\n\n## Second\nWorld", "folder/test.md"
        )
        self.assertEqual([chunk["text"] for chunk in chunks], [
            "# Title", "## First\nHello", "## Second\nWorld"
        ])
        self.assertTrue(all(chunk["pages"] == [] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()