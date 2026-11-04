"""Local HTTP entry point for ingesting the seven Zava sample documents."""

import logging
from functools import partial
from threading import Lock
from typing import Literal, get_args

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from fastapi import FastAPI, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field
from pypdf.errors import PdfReadError

from graphrag.build_graph import build_graph_for_chunks
from graphrag.components import ZavaDocumentParser, ZavaDocumentSplitter
from graphrag.config import PROJECT_ROOT
from graphrag.database_mcp.client import DatabaseMcpClient, DatabaseMcpError
from graphrag.documents import (
    create_chunks,
    create_markdown_chunks,
    parse_pdf_document,
)


logger = logging.getLogger(__name__)
DOCUMENT_DIR = PROJECT_ROOT / "document_storage"
ingestion_lock = Lock()

DocumentName = Literal[
    "Benefit_Options.pdf",
    "Northwind_Health_Plus_Benefits_Details.pdf",
    "Northwind_Standard_Benefits_Details.pdf",
    "PerksPlus.pdf",
    "Zava_Company_Overview.md",
    "employee_handbook.pdf",
    "role_library.pdf",
]


class DocumentInfo(BaseModel):
    filename: DocumentName
    available: bool


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: DocumentName


class IngestResponse(BaseModel):
    status: Literal["completed"] = "completed"
    document: DocumentName
    chunks: int = Field(gt=0)
    entity_mentions: int = Field(ge=0)
    relationship_mentions: int = Field(ge=0)


app = FastAPI(
    title="Zava GraphRAG Document Ingestion",
    description=(
        "Ingest one of the seven local Zava documents into the existing graph. "
        "Local use only; requests wait for processing to finish."
    ),
)


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    root = DOCUMENT_DIR.resolve()
    return [
        DocumentInfo(
            filename=name,
            available=(
                (root / name).resolve().parent == root
                and (root / name).is_file()
            ),
        )
        for name in get_args(DocumentName)
    ]


@app.post("/documents/ingest", response_model=IngestResponse)
def ingest_document(body: IngestRequest) -> IngestResponse:
    if not ingestion_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another ingestion is running. Wait for it to finish.",
        )

    try:
        root = DOCUMENT_DIR.resolve()
        document_path = (root / body.filename).resolve()
        if document_path.parent != root:
            raise HTTPException(
                status_code=400,
                detail="The document must be inside local document storage.",
            )

        try:
            document_bytes = document_path.read_bytes()
        except FileNotFoundError:
            logger.error("Ingestion document was not found: %s.", body.filename)
            raise HTTPException(
                status_code=404,
                detail="Document was not found in local document storage.",
            ) from None
        if not document_bytes.strip():
            raise HTTPException(status_code=422, detail="The document is empty.")

        with DefaultAzureCredential() as credential:
            with DatabaseMcpClient() as client:
                client.health_check()
                client.ensure_schema()

                parser = ZavaDocumentParser(
                    pdf_parser=partial(parse_pdf_document, credential=credential),
                )
                splitter = ZavaDocumentSplitter(
                    pdf_splitter=create_chunks,
                    markdown_splitter=create_markdown_chunks,
                )
                parsed = parser.parse_document(document_bytes, body.filename)
                chunks = splitter.split_document(parsed)
                if not chunks:
                    raise HTTPException(
                        status_code=422,
                        detail="The document contains no usable text.",
                    )

                counts = build_graph_for_chunks(
                    client, [chunk.to_record() for chunk in chunks],
                )

        return IngestResponse(document=body.filename, **counts)
    except (PdfReadError, UnicodeDecodeError) as exc:
        logger.error("Document parsing failed for %s (%s).", body.filename, type(exc).__name__)
        raise HTTPException(
            status_code=422,
            detail="The document is not a readable PDF or UTF-8 Markdown file.",
        ) from None
    except (DatabaseMcpError, AzureError, OpenAIError, OSError) as exc:
        logger.error("Ingestion service failed for %s (%s).", body.filename, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail=(
                "Ingestion failed. Check local file access, Azure sign-in and model "
                "settings, and the database MCP server. Earlier writes may remain."
            ),
        ) from None
    except (ValueError, RuntimeError) as exc:
        logger.error("Ingestion processing failed for %s (%s).", body.filename, type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail=(
                "Ingestion could not complete parsing or a validated graph build. "
                "Check document content, configuration and service logs. "
                "Earlier writes may remain."
            ),
        ) from None
    finally:
        ingestion_lock.release()
