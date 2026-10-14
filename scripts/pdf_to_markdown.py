
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from io import BytesIO

from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat


ENDPOINT = "https://graphrag.cognitiveservices.azure.com/"

PDF_PATH = Path("role_library.pdf")
OUTPUT_PATH = Path("data/markdown/role_library.md")

BATCH_SIZE = 2


def main():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))
    total_pages = len(reader.pages)

    print(f"Total PDF pages: {total_pages}")

    credential = DefaultAzureCredential()

    try:
        client = DocumentIntelligenceClient(
            endpoint=ENDPOINT,
            credential=credential,
        )

        markdown_parts = []

        for start in range(0, total_pages, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_pages)

            print(f"Processing pages {start + 1}-{end}...")

            writer = PdfWriter()

            for page_number in range(start, end):
                writer.add_page(reader.pages[page_number])

            pdf_buffer = BytesIO()
            writer.write(pdf_buffer)
            pdf_buffer.seek(0)

            poller = client.begin_analyze_document(
                "prebuilt-layout",
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

            print(
                f"Extracted {len(extracted_text)} characters "
                f"from pages {start + 1}-{end}"
            )

            markdown_parts.append(extracted_text)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        OUTPUT_PATH.write_text(
            "\n\n".join(markdown_parts),
            encoding="utf-8",
        )

        print("\nConversion completed successfully!")
        print(f"Processed pages: {total_pages}")
        print(f"Markdown saved to: {OUTPUT_PATH.resolve()}")

    finally:
        credential.close()


if __name__ == "__main__":
    main()
