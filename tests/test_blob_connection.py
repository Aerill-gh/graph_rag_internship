from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient
from pathlib import Path

from graphrag.config import settings

def main():
    credential = AzureCliCredential()

    try:
        client = BlobServiceClient(
            account_url=settings.storage_account_url,
            credential=credential,
        )

        blob_client = client.get_blob_client(
            container=settings.blob_container,
            blob=settings.blob_name,
        )

        output_path = Path(settings.blob_name).name

        with open(output_path, "wb") as file:
            file.write(blob_client.download_blob().readall())

        print(f"Downloaded successfully: {output_path}")
        print(f"Saved to: {Path(output_path).resolve()}")

        client.close()

    finally:
        credential.close()


if __name__ == "__main__":
    main()