from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient
from pathlib import Path

STORAGE_ACCOUNT_NAME = "aidepartmentstorage"
CONTAINER_NAME = "docs"

# Replace with the exact blob name shown in your container
BLOB_NAME = "role_library.pdf"

def main():
    account_url = (
        f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    )

    credential = AzureCliCredential()

    try:
        client = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )

        blob_client = client.get_blob_client(
            container=CONTAINER_NAME,
            blob=BLOB_NAME,
        )

        output_path = Path(BLOB_NAME).name

        with open(output_path, "wb") as file:
            file.write(blob_client.download_blob().readall())

        print(f"Downloaded successfully: {output_path}")
        print(f"Saved to: {Path(output_path).resolve()}")

        client.close()

    finally:
        credential.close()


if __name__ == "__main__":
    main()