import hashlib
import json

from openai import OpenAI

from graphrag.config import PROJECT_ROOT, settings


CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "entity_extraction"

SYSTEM_PROMPT = """
You are an entity extraction assistant for the Zava GraphRAG project.

Extract information strictly from the provided document chunk and figure captions.

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
6. Include only roles whose names are explicitly stated in this chunk.
7. Return an empty roles list when no role is explicitly identified.
8. Return valid JSON only.

JSON format:
{
    "roles": [
        {
            "name": "Role name",
            "skills": ["Skill 1", "Skill 2"],
            "department": null
        }
    ]
}
"""

MODEL_SETTINGS = {
    "max_output_tokens": 1500,
    "temperature": 0,
    "text": {"format": {"type": "json_object"}},
}


def validate_result(result):
    if not isinstance(result, dict):
        raise ValueError("Extraction result must be a JSON object.")

    if not isinstance(result.get("roles"), list):
        raise ValueError("Roles must be a list.")

    for role in result["roles"]:
        if not isinstance(role, dict) or not isinstance(role.get("name"), str):
            raise ValueError("Every role must have a string name.")

        if not isinstance(role.get("skills"), list):
            raise ValueError("Role skills must be a list.")

        if not all(isinstance(skill, str) for skill in role["skills"]):
            raise ValueError("Every skill must be a string.")

        department = role.get("department")
        if department is not None and not isinstance(department, str):
            raise ValueError("Department must be a string or null.")


def build_request_input(chunk):
    pages = ", ".join(str(page) for page in chunk.get("pages", [])) or "unknown"
    figure_captions = [
        figure["caption"].strip()
        for figure in chunk.get("figures", [])
        if figure.get("caption", "").strip()
    ]
    figure_context = "\n".join(figure_captions) or "No figure captions available."

    return (
        "Extract entities from this document chunk and return a valid JSON object.\n"
        f"Source document: {chunk['source_document']}\n"
        f"Pages: {pages}\n\n"
        f"Source text:\n{chunk['text']}\n\n"
        f"Figure captions:\n{figure_context}"
    )


def extract_entities(chunk):
    azure_openai = settings.azure_openai

    request_input = build_request_input(chunk)

    cache_input = {
        "deployment": azure_openai.deployment,
        "base_url": azure_openai.base_url,
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
        print("Using cached entity extraction")
        result = json.loads(cache_path.read_text(encoding="utf-8"))
        validate_result(result)
        return result

    client = OpenAI(
        api_key=azure_openai.api_key,
        base_url=azure_openai.base_url,
    )

    try:
        response = client.responses.create(
            model=azure_openai.deployment,
            instructions=SYSTEM_PROMPT,
            input=request_input,
            **MODEL_SETTINGS,
        )
    finally:
        client.close()

    if not response.output_text:
        raise ValueError("GPT-4.1 returned an empty response.")

    result = json.loads(response.output_text)
    validate_result(result)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Entity extraction cached")
    return result
