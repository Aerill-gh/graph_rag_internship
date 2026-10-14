import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url=os.environ["AZURE_OPENAI_BASE_URL"],
)

response = client.responses.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    input="Reply with exactly: GPT-4.1 connection successful",
    max_output_tokens=50,
)

print(response.output_text)
