import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


INPUT_PATH = Path("data/chunks/role_library_chunks.json")
OUTPUT_PATH = Path("data/extracted/test_extraction.json")
CACHE_DIR = Path("data/cache/entity_extraction")

SYSTEM_PROMPT = """
You are an entity extraction assistant for the Zava GraphRAG project.

Extract information strictly from the provided document chunk.

Follow the Zava ontology:
- Role
- Skill
- Department

Allowed relationships:
- Role -[REQUIRES]-> Skill
- Role -[BELONGS_TO]-> Department

Rules:
1. Do not invent entities or relationships.
2. Only extract facts supported by the source text.
3. Extract only explicitly required skills, competencies,
   abilities, and knowledge from the Qualifications section.

   Do not extract job responsibilities, duties, tasks,
   education degrees, years of experience, or employment history.

   Do not infer skills from the Responsibilities section.
4. Extract the department only if explicitly stated.
5. Do not infer a department from the role title.
6. Preserve the role name provided in the chunk.
7. Return valid JSON only.

JSON format:
{
  "role": "Role name",
  "skills": ["Skill 1", "Skill 2"],
  "department": null
}
"""

MODEL_SETTINGS = {
    "max_output_tokens": 1500,
    "temperature": 0,
    "text": {"format": {"type": "json_object"}},
}


def validate_result(result, chunk):
    if not isinstance(result, dict):
        raise ValueError("Extraction result must be a JSON object.")

    if result.get("role") != chunk["role"]:
        raise ValueError("Extracted role does not match the source.")

    if not isinstance(result.get("skills"), list):
        raise ValueError("Skills must be a list.")

    if not all(isinstance(skill, str) for skill in result["skills"]):
        raise ValueError("Every skill must be a string.")

    if result.get("department") is not None and not isinstance(
        result["department"], str
    ):
        raise ValueError("Department must be a string or null.")


def extract_entities(chunk):
    load_dotenv()

    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    base_url = os.environ["AZURE_OPENAI_BASE_URL"]

    request_input = (
        "Extract entities from the following document. "
        "Return the result as a valid JSON object.\n\n"
        f"Role: {chunk['role']}\n\n"
        f"Source text:\n{chunk['text']}"
    )

    cache_input = {
        "deployment": deployment,
        "base_url": base_url,
        "system_prompt": SYSTEM_PROMPT,
        "input": request_input,
        "settings": MODEL_SETTINGS,
    }

    cache_key = hashlib.sha256(
        json.dumps(
            cache_input,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.is_file():
        print("Using cached GPT-4.1 extraction")
        result = json.loads(cache_path.read_text(encoding="utf-8"))
        validate_result(result, chunk)
        return result

    client = OpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=base_url,
    )

    try:
        response = client.responses.create(
            model=deployment,
            instructions=SYSTEM_PROMPT,
            input=request_input,
            **MODEL_SETTINGS,
        )
    finally:
        client.close()

    if not response.output_text:
        raise ValueError("GPT-4.1 returned an empty response.")

    result = json.loads(response.output_text)
    validate_result(result, chunk)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("GPT-4.1 extraction cached")
    return result


def main():
    chunks = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    if not chunks:
        raise ValueError("No chunks found in the input file.")

    # Test only the first chunk
    chunk = chunks[0]

    print(f"Extracting: {chunk['role']}")

    result = extract_entities(chunk)

    output = {
        "chunk_id": chunk["chunk_id"],
        "source_document": chunk["source_document"],
        "entities": result,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
