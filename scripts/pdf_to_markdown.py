import hashlib
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat


load_dotenv()
ENDPOINT = os.environ["DOCINTEL_ENDPOINT"]

PDF_PATH = Path("role_library.pdf")
OUTPUT_PATH = Path("data/markdown/role_library.md")
CACHE_DIR = Path("data/cache/document_intelligence")

BATCH_SIZE = 2
MODEL_ID = "prebuilt-layout"
PAGE_BREAK = "<!-- PageBreak -->"


def main():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pdf_hash = hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()
    reader = PdfReader(str(PDF_PATH))
    total_pages = len(reader.pages)

    print(f"Total PDF pages: {total_pages}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    markdown_parts = []

    credential = None

    try:
        for start in range(0, total_pages, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_pages)
            batch_pages = end - start

            cache_key = hashlib.sha256(
                f"{pdf_hash}:{start}:{end}:{MODEL_ID}:markdown:{ENDPOINT}".encode()
            ).hexdigest()
            cache_path = CACHE_DIR / f"{cache_key}.md"

            if cache_path.is_file():
                extracted_text = cache_path.read_text(encoding="utf-8")
                print(f"Using cached pages {start + 1}-{end}")
            else:
                print(f"Processing pages {start + 1}-{end}...")

                if credential is None:
                    credential = DefaultAzureCredential()
                    client = DocumentIntelligenceClient(
                        endpoint=ENDPOINT,
                        credential=credential,
                    )

                writer = PdfWriter()
                for page_number in range(start, end):
                    writer.add_page(reader.pages[page_number])

                pdf_buffer = BytesIO()
                writer.write(pdf_buffer)
                pdf_buffer.seek(0)

                poller = client.begin_analyze_document(
                    MODEL_ID,
                    body=pdf_buffer,
                    output_content_format=DocumentContentFormat.MARKDOWN,
                )

                result = poller.result()
                extracted_text = (result.content or "").strip()

                if not extracted_text or not any(
                    char.isalnum() for char in extracted_text
                ):
                    raise RuntimeError(
                        f"No meaningful content extracted from pages "
                        f"{start + 1}-{end}"
                    )

                cache_path.write_text(extracted_text, encoding="utf-8")
                print(f"Cached pages {start + 1}-{end}")

            if not extracted_text.strip() or not any(
                char.isalnum() for char in extracted_text
            ):
                raise RuntimeError(
                    f"Invalid cached content for pages {start + 1}-{end}"
                )

            # Verify page breaks inside each batch.
            expected_breaks = batch_pages - 1
            actual_breaks = extracted_text.count(PAGE_BREAK)

            if actual_breaks != expected_breaks:
                raise RuntimeError(
                    f"Page-break mismatch for pages {start + 1}-{end}: "
                    f"expected {expected_breaks}, found {actual_breaks}. "
                    f"Check Document Intelligence Markdown output."
                )

            markdown_parts.append(extracted_text)

        # Preserve page boundaries between batches.
        markdown = (
            f"\n\n{PAGE_BREAK}\n\n".join(markdown_parts)
        )

        # A PDF with N pages should have N-1 page breaks.
        expected_total_breaks = total_pages - 1
        actual_total_breaks = markdown.count(PAGE_BREAK)

        if actual_total_breaks != expected_total_breaks:
            raise RuntimeError(
                f"Total page-break mismatch: "
                f"expected {expected_total_breaks}, "
                f"found {actual_total_breaks}"
            )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(markdown, encoding="utf-8")

        print("\nConversion completed successfully!")
        print(f"Processed pages: {total_pages}")
        print(f"Page-break markers: {actual_total_breaks}")
        print(f"Markdown saved to: {OUTPUT_PATH.resolve()}")

    finally:
        if credential is not None:
            credential.close()


if __name__ == "__main__":
    main()
