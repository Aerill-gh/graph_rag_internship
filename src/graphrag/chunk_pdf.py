import argparse
import json
from pathlib import PurePosixPath

from azure.identity import DefaultAzureCredential

from graphrag.components import LocalDocumentLoader
from graphrag.documents import (
    create_markdown_chunks,
    create_pdf_chunks,
    output_path_for_document,
)


def process_document(document_name, credential):
    document_bytes = LocalDocumentLoader().load(document_name)
    source_document = PurePosixPath(document_name.replace("\\", "/")).as_posix()

    suffix = PurePosixPath(source_document).suffix.lower()
    if suffix == ".pdf":
        chunks, figure_count = create_pdf_chunks(
            document_bytes,
            source_document,
            credential,
        )
    elif suffix in {".md", ".markdown"}:
        chunks = create_markdown_chunks(document_bytes, source_document)
        figure_count = 0
    else:
        raise ValueError(f"Unsupported document type: {source_document}")

    output_path = output_path_for_document(source_document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Chunks created: {len(chunks)}")
    print(f"Figures saved: {figure_count}")
    print(f"Output: {output_path}")


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Chunk local documents from document_storage."
    )
    parser.add_argument(
        "--document",
        dest="document_names",
        action="append",
        metavar="RELATIVE_PATH",
        help="Path relative to document_storage; repeat for multiple documents.",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    document_names = args.document_names or ["role_library.pdf"]
    credential = DefaultAzureCredential()
    try:
        for document_name in document_names:
            print(f"Processing local document: {document_name}")
            process_document(document_name, credential)
    finally:
        credential.close()


if __name__ == "__main__":
    main()
