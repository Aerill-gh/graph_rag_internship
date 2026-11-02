from openai import OpenAI

from graphrag.config import settings


def main():
    azure_openai = settings.azure_openai

    client = OpenAI(
        api_key=azure_openai.api_key,
        base_url=azure_openai.base_url,
    )

    response = client.responses.create(
        model=azure_openai.deployment,
        input="Reply with exactly: GPT-4.1 connection successful",
        max_output_tokens=50,
    )

    print(response.output_text)


if __name__ == "__main__":
    main()