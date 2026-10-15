import json
import re
from pathlib import Path

import tiktoken


INPUT_PATH = Path("data/markdown/role_library_cleaned.md")
OUTPUT_PATH = Path("data/chunks/role_library_chunks.json")

SOURCE_DOCUMENT = "role_library.pdf"
MAX_TOKENS = 500
OVERLAP_TOKENS = 50

encoder = tiktoken.get_encoding("cl100k_base")


def token_count(text):
    return len(encoder.encode(text))


def split_large_text(text):
    tokens = encoder.encode(text)
    step = MAX_TOKENS - OVERLAP_TOKENS

    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + MAX_TOKENS]

        if not chunk_tokens:
            break

        yield encoder.decode(chunk_tokens)

        if start + MAX_TOKENS >= len(tokens):
            break


def extract_roles(markdown):
    pattern = r"(?m)^## (.+)$"
    matches = list(re.finditer(pattern, markdown))

    roles = []

    for index, match in enumerate(matches):
        role_name = match.group(1).strip()

        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(markdown)
        )

        content = markdown[start:end].strip()

        # Page number where the role begins
        start_page = markdown[:match.start()].count(
            "<!-- PageBreak -->"
        ) + 1

        roles.append({
            "role": role_name,
            "content": content,
            "start_page": start_page,
        })

    return roles



def create_chunks(roles):
    chunks = []
    page_marker = "<!-- PageBreak -->"

    for role in roles:
        role_name = role["role"]
        start_page = role["start_page"]

        # Keep page markers temporarily to track source pages.
        content = role["content"]
        page_parts = content.split(page_marker)

        # Tokenize each page separately, preserving page metadata.
        token_pages = []
        for page_offset, page_text in enumerate(page_parts):
            page_number = start_page + page_offset
            tokens = encoder.encode(page_text.strip())

            token_pages.extend(
                (token, page_number) for token in tokens
            )

        prefix = f"Role: {role_name}\n\n"
        available_tokens = MAX_TOKENS - token_count(prefix)

        if available_tokens <= OVERLAP_TOKENS:
            raise ValueError("Role prefix is too long.")

        step = available_tokens - OVERLAP_TOKENS

        for start in range(0, len(token_pages), step):
            chunk_data = token_pages[start:start + available_tokens]

            if not chunk_data:
                break

            chunk_tokens = [token for token, _ in chunk_data]
            pages = sorted(set(page for _, page in chunk_data))

            text = prefix + encoder.decode(chunk_tokens)

            chunks.append({
                "chunk_id": f"role_library_{len(chunks) + 1:03d}",
                "source_document": SOURCE_DOCUMENT,
                "role": role_name,
                "pages": pages,
                "text": text,
                "token_count": token_count(text),
            })

            if start + available_tokens >= len(token_pages):
                break

    return chunks



def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)

    markdown = INPUT_PATH.read_text(encoding="utf-8")

    roles = extract_roles(markdown)
    chunks = create_chunks(roles)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Roles found: {len(roles)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Maximum tokens: {max(c['token_count'] for c in chunks)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
