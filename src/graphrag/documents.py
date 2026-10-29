import hashlib
import json
import re
from io import BytesIO
from pathlib import Path, PurePosixPath

import tiktoken
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from pypdf import PdfReader, PdfWriter
 
from graphrag.config import settings

OUTPUT_DIR = Path("data/chunks")
CACHE_DIR = Path("data/cache/document_intelligence/chunking")
FIGURE_DIR = CACHE_DIR / "figures"
 
BATCH_SIZE = 2
MODEL_ID = "prebuilt-layout"
MAX_TOKENS = 500
OVERLAP_TOKENS = 50
 
encoder = tiktoken.get_encoding("cl100k_base")
 
 
def analyze_batch(client, reader, pdf_hash, start, end):
    cache_key = hashlib.sha256(
        f"{pdf_hash}:{start}:{end}:{MODEL_ID}:text+figures:{settings.docintel_endpoint}".encode()
    ).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.json"
 
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(Path(figure["image_path"]).is_file() for figure in cached["figures"]):
            print(f"Using cached pages {start + 1}-{end}")
            return cached
 
    print(f"Analyzing pages {start + 1}-{end}...")
    writer = PdfWriter()
    for page_number in range(start, end):
        writer.add_page(reader.pages[page_number])
 
    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)
    pdf_buffer.seek(0)
 
    poller = client.begin_analyze_document(
        MODEL_ID,
        body=pdf_buffer,
        output_content_format=DocumentContentFormat.TEXT,
        output=["figures"],
    )
    result = poller.result()
    result_id = poller.details["operation_id"]
 
    pages = []
    for page in result.pages or []:
        page_text = "\n".join(line.content for line in (page.lines or []))
        pages.append({
            "page": start + page.page_number,
            "text": page_text.strip(),
        })
 
    figures = []
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
 
    if not pages:
        raise RuntimeError(f"No pages returned for pages {start + 1}-{end}")
 
    cached = {"pages": pages, "figures": figures}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    return cached
 
 
def chunk_prefix_for_blob(blob_name):
    blob_path = PurePosixPath(blob_name.replace("\\", "/").strip("/"))
    return "_".join(blob_path.with_suffix("").parts)
 
 
def output_path_for_blob(blob_name):
    blob_path = PurePosixPath(blob_name.replace("\\", "/").strip("/"))
    if not blob_path.name or any(part == ".." for part in blob_path.parts):
        raise ValueError(f"Invalid Blob name: {blob_name}")
 
    output_path = blob_path.with_name(f"{blob_path.stem}_chunks.json")
    return OUTPUT_DIR.joinpath(*output_path.parts)
 
 
def create_chunks(pages, figures, source_document):
    figures_by_page = {}
    for figure in figures:
        figures_by_page.setdefault(figure["page"], []).append(figure)
 
    page_tokens = []
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
 
    if not page_tokens:
        raise RuntimeError("The document contains no chunkable text.")
 
    chunks = []
    chunk_prefix = chunk_prefix_for_blob(source_document)
    step = MAX_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(page_tokens), step):
        chunk_data = page_tokens[start:start + MAX_TOKENS]
        if not chunk_data:
            break
 
        chunk_pages = sorted({page for _, page in chunk_data if page is not None})
        chunk_figures = [
            figure
            for figure in figures
            if figure["page"] in chunk_pages
        ]
        text = encoder.decode([token for token, _ in chunk_data])
 
        chunks.append({
            "chunk_id": f"{chunk_prefix}_{len(chunks) + 1:03d}",
            "source_document": source_document,
            "pages": chunk_pages,
            "text": text,
            "token_count": len(encoder.encode(text)),
            "figures": chunk_figures,
        })
 
        if start + MAX_TOKENS >= len(page_tokens):
            break
 
    return chunks
 
 
def parse_pdf_document(pdf_bytes, credential):
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    reader = PdfReader(BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    print(f"Total PDF pages: {total_pages}")
 
    client = None
    try:
        client = DocumentIntelligenceClient(
            endpoint=settings.docintel_endpoint,
            credential=credential,
        )
 
        pages = []
        figures = []
        for start in range(0, total_pages, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_pages)
            batch = analyze_batch(client, reader, pdf_hash, start, end)
            pages.extend(batch["pages"])
            figures.extend(batch["figures"])
 
        return {"pages": pages, "figures": figures}
    finally:
        if client is not None:
            client.close()
 
 
def create_pdf_chunks(pdf_bytes, blob_name, credential):
    parsed = parse_pdf_document(pdf_bytes, credential)
    chunks = create_chunks(parsed["pages"], parsed["figures"], blob_name)
    return chunks, len(parsed["figures"])


def create_markdown_chunks(markdown_bytes, blob_name):
    text = markdown_bytes.decode("utf-8-sig").strip()
    sections = re.split(r"(?m)(?=^#{1,6}\s)", text)
    chunks = []
    prefix = chunk_prefix_for_blob(blob_name)
    for section in sections:
        section = section.strip()
        if not section:
            continue
        section_chunks = create_chunks([{"page": None, "text": section}], [], blob_name)
        for chunk in section_chunks:
            chunk["chunk_id"] = f"{prefix}_{len(chunks) + 1:03d}"
            chunks.append(chunk)
    return chunks

