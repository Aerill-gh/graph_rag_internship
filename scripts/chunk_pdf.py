import re
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TypedDict

import tiktoken
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentContentFormat
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
from neo4j import Session
from pypdf import PdfReader, PdfWriter

from graphrag.embeddings import get_chunk_embedding
from graphrag.neo4j_store import create_neo4j_driver, store_chunk
from graphrag.config import settings

CACHE_DIR = Path("data/cache/document_intelligence/chunking")
FIGURE_DIR = CACHE_DIR / "figures"

BATCH_SIZE = 2
MODEL_ID = "prebuilt-layout"
MAX_TOKENS = 500
OVERLAP_TOKENS = 50

encoder = tiktoken.get_encoding("cl100k_base")


class PageData(TypedDict):
    page: int
    text: str


class FigureData(TypedDict):
    id: str
    page: int
    caption: str
    image_path: str


class BatchData(TypedDict):
    pages: list[PageData]
    figures: list[FigureData]


class ChunkData(TypedDict):
    chunk_id: str
    source_document: str
    pages: list[int]
    text: str
    token_count: int
    figures: list[FigureData]


def batch_cache_path(pdf_hash: str, start: int, end: int) -> Path:
    cache_key = hashlib.sha256(
        f"{pdf_hash}:{start}:{end}:{MODEL_ID}:text+figures:{settings.docintel_endpoint}".encode()
    ).hexdigest()
    return CACHE_DIR / f"{cache_key}.json"


def load_cached_batch(cache_path: Path) -> BatchData | None:
    if cache_path.is_file():
        cached: BatchData = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(Path(figure["image_path"]).is_file() for figure in cached["figures"]):
            return cached
    return None


def save_cached_batch(cache_path: Path, batch: BatchData) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")


def create_batch_pdf(reader: PdfReader, start: int, end: int) -> BytesIO:
    writer = PdfWriter()
    for page_number in range(start, end):
        writer.add_page(reader.pages[page_number])

    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer


def extract_pages(result: AnalyzeResult, start: int) -> list[PageData]:
    pages: list[PageData] = []
    for page in result.pages or []:
        page_text = "\n".join(line.content for line in (page.lines or []))
        pages.append({
            "page": start + page.page_number,
            "text": page_text.strip(),
        })
    return pages


def download_figures(
    client: DocumentIntelligenceClient,
    result: AnalyzeResult,
    result_id: str,
    pdf_hash: str,
    start: int,
    end: int,
) -> list[FigureData]:
    figures: list[FigureData] = []
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for figure in result.figures or []:
        local_page = (
            figure.bounding_regions[0].page_number
            if figure.bounding_regions
            else None
        )
        if local_page is None:
            continue

        figure_id = str(figure.id)
        image_path = FIGURE_DIR / (
            f"{pdf_hash}_{start + 1}-{end}_{figure_id.replace('.', '_')}.png"
        )
        image_bytes = b"".join(
            client.get_analyze_result_figure(MODEL_ID, result_id, figure_id)
        )
        image_path.write_bytes(image_bytes)

        figures.append({
            "id": figure_id,
            "page": start + local_page,
            "caption": figure.caption.content if figure.caption else "",
            "image_path": image_path.as_posix(),
        })
    return figures


def analyze_batch(
    client: DocumentIntelligenceClient,
    reader: PdfReader,
    pdf_hash: str,
    start: int,
    end: int,
) -> BatchData:
    cache_path = batch_cache_path(pdf_hash, start, end)
    cached = load_cached_batch(cache_path)
    if cached is not None:
        print(f"Using cached pages {start + 1}-{end}")
        return cached

    print(f"Analyzing pages {start + 1}-{end}...")
    poller = client.begin_analyze_document(
        MODEL_ID,
        body=create_batch_pdf(reader, start, end),
        output_content_format=DocumentContentFormat.TEXT,
        output=["figures"],
    )
    result = poller.result()
    pages = extract_pages(result, start)
    figures = download_figures(
        client, result, poller.details["operation_id"], pdf_hash, start, end
    )

    if not pages:
        raise RuntimeError(f"No pages returned for pages {start + 1}-{end}")

    batch: BatchData = {"pages": pages, "figures": figures}
    save_cached_batch(cache_path, batch)
    return batch


def chunk_prefix_for_blob(blob_name: str) -> str:
    blob_path = PurePosixPath(blob_name.replace("\\", "/").strip("/"))
    return "_".join(blob_path.with_suffix("").parts)


def tokenize_pages(
    pages: list[PageData], figures: list[FigureData]
) -> list[tuple[int, int]]:
    figures_by_page: dict[int, list[FigureData]] = {}
    for figure in figures:
        figures_by_page.setdefault(figure["page"], []).append(figure)

    page_tokens: list[tuple[int, int]] = []
    for page in pages:
        text = page["text"]
        if not text and figures_by_page.get(page["page"]):
            captions = [
                figure["caption"]
                for figure in figures_by_page[page["page"]]
                if figure["caption"]
            ]
            text = "\n".join(captions) or f"Figure on page {page['page']}."

        page_tokens.extend(
            (token, page["page"])
            for token in encoder.encode(text)
        )
    return page_tokens


def create_chunk_from_tokens(
    chunk_tokens: list[tuple[int, int]],
    figures: list[FigureData],
    source_document: str,
    chunk_id: str,
) -> ChunkData:
    chunk_pages = sorted({page for _, page in chunk_tokens if page is not None})
    chunk_figures = [
        figure
        for figure in figures
        if figure["page"] in chunk_pages
    ]
    text = encoder.decode([token for token, _ in chunk_tokens])

    return {
        "chunk_id": chunk_id,
        "source_document": source_document,
        "pages": chunk_pages,
        "text": text,
        "token_count": len(encoder.encode(text)),
        "figures": chunk_figures,
    }


def create_chunks(
    pages: list[PageData], figures: list[FigureData], source_document: str
) -> list[ChunkData]:
    page_tokens = tokenize_pages(pages, figures)

    if not page_tokens:
        raise RuntimeError("The document contains no chunkable text.")

    chunks: list[ChunkData] = []
    chunk_prefix = chunk_prefix_for_blob(source_document)
    step = MAX_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(page_tokens), step):
        chunk_data = page_tokens[start:start + MAX_TOKENS]
        if not chunk_data:
            break

        chunks.append(create_chunk_from_tokens(
            chunk_data,
            figures,
            source_document,
            f"{chunk_prefix}_{len(chunks) + 1:03d}",
        ))

        if start + MAX_TOKENS >= len(page_tokens):
            break

    return chunks


def parse_pdf_document(
    pdf_bytes: bytes,
    credential: DefaultAzureCredential,
) -> BatchData:
    """Extract PDF pages and figures using Azure Document Intelligence."""
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    reader = PdfReader(BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    print(f"Total PDF pages: {total_pages}")

    client = DocumentIntelligenceClient(
        endpoint=settings.docintel_endpoint,
        credential=credential,
    )

    try:
        pages: list[PageData] = []
        figures: list[FigureData] = []

        for start in range(0, total_pages, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_pages)
            batch = analyze_batch(client, reader, pdf_hash, start, end)

            pages.extend(batch["pages"])
            figures.extend(batch["figures"])

        return {"pages": pages, "figures": figures}
    finally:
        client.close()



def create_pdf_chunks(
    pdf_bytes: bytes,
    blob_name: str,
    credential: DefaultAzureCredential,
) -> tuple[list[ChunkData], int]:
    """Parse a PDF and split its content into chunks."""
    parsed_data = parse_pdf_document(pdf_bytes, credential)

    chunks = create_chunks(
        parsed_data["pages"],
        parsed_data["figures"],
        blob_name,
    )

    return chunks, len(parsed_data["figures"])



def create_markdown_chunks(markdown_bytes: bytes, blob_name: str) -> list[ChunkData]:
    text = markdown_bytes.decode("utf-8-sig")
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    chunks: list[ChunkData] = []
    chunk_prefix = chunk_prefix_for_blob(blob_name)

    for section in sections:
        section = section.strip()

        if not section:
            continue

        chunks.append({
            "chunk_id": f"{chunk_prefix}_{len(chunks) + 1:03d}",
            "source_document": blob_name,
            "pages": [],
            "text": section,
            "token_count": len(section.split()),
            "figures": [],
        })

    return chunks


def download_blob_bytes(blob_name: str, credential: DefaultAzureCredential) -> bytes:
    with BlobClient(
        account_url=settings.storage_account_url,
        container_name=settings.blob_container,
        blob_name=blob_name,
        credential=credential,
    ) as blob_client:
        return blob_client.download_blob().readall()


def create_document_chunks(
    document_bytes: bytes, blob_name: str, credential: DefaultAzureCredential
) -> tuple[list[ChunkData], int]:
    suffix = PurePosixPath(blob_name.replace("\\", "/")).suffix.lower()

    if suffix == ".pdf":
        return create_pdf_chunks(
            document_bytes,
            blob_name,
            credential,
        )
    elif suffix in {".md", ".markdown"}:
        return create_markdown_chunks(document_bytes, blob_name), 0
    else:
        raise ValueError(f"Unsupported document type for Blob: {blob_name}")


def embed_and_store_chunks(session: Session, chunks: list[ChunkData]) -> None:
    for chunk in chunks:
        embedding = get_chunk_embedding(chunk)
        store_chunk(session, chunk, embedding)


def process_blob(
    blob_name: str, credential: DefaultAzureCredential, session: Session
) -> None:
    document_bytes = download_blob_bytes(blob_name, credential)
    chunks, figure_count = create_document_chunks(document_bytes, blob_name, credential)
    embed_and_store_chunks(session, chunks)

    print(f"Chunks stored in Neo4j: {len(chunks)}")
    print(f"Figures processed: {figure_count}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk selected documents directly from Azure Blob Storage."
    )
    parser.add_argument(
        "--blob",
        dest="blob_names",
        action="append",
        metavar="BLOB_NAME",
        help="Blob to process; repeat this option to process multiple documents.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    blob_names = args.blob_names or [settings.blob_name]

    credential = DefaultAzureCredential()

    try:
        with create_neo4j_driver() as driver:
            driver.verify_connectivity()

            with driver.session(database="neo4j") as session:
                for blob_name in blob_names:
                    print(f"Processing Blob: {blob_name}")
                    process_blob(blob_name, credential, session)
    finally:
        credential.close()


if __name__ == "__main__":
    main()