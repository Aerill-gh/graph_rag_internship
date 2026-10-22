import argparse
import json
from pathlib import PurePosixPath

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient

from graphrag.config import settings
from graphrag.documents import (
    create_markdown_chunks,
    create_pdf_chunks,
    output_path_for_blob,
)


def process_blob(blob_name, credential):
    with BlobClient(
        account_url=settings.storage_account_url,
        container_name=settings.blob_container,
        blob_name=blob_name,
        credential=credential,
    ) as blob_client:
        document_bytes = blob_client.download_blob().readall()

    suffix = PurePosixPath(blob_name.replace("\\", "/")).suffix.lower()
    if suffix == ".pdf":
        chunks, figure_count = create_pdf_chunks(
            document_bytes,
            blob_name,
            credential,
        )
    elif suffix in {".md", ".markdown"}:
        chunks = create_markdown_chunks(document_bytes, blob_name)
        figure_count = 0
    else:
        raise ValueError(f"Unsupported document type for Blob: {blob_name}")

    output_path = output_path_for_blob(blob_name)
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


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    blob_names = args.blob_names or [settings.blob_name]
    credential = DefaultAzureCredential()
    try:
        for blob_name in blob_names:
            print(f"Processing Blob: {blob_name}")
            process_blob(blob_name, credential)
    finally:
        credential.close()


if __name__ == "__main__":
    main()
