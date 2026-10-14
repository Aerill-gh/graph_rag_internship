import re
from pathlib import Path

INPUT_PATH = Path("data/markdown/role_library.md")
OUTPUT_PATH = Path("data/markdown/role_library_cleaned.md")


def clean_markdown(text):
    # Normalize line endings.
    text = text.replace("\r\n", "\n")

    # Preserve page boundaries for future source tracking.
    text = re.sub(
        r"<!--\s*PageBreak\s*-->",
        "\n\n<!-- PageBreak -->\n\n",
        text,
    )

    # Normalize bullet characters.
    text = re.sub(
        r"(?m)^[ \t]*[·•][ \t]*",
        "- ",
        text,
    )

    # Normalize escaped Markdown bullets.
    text = re.sub(
        r"(?m)^[ \t]*\\-[ \t]+",
        "- ",
        text,
    )

    # Convert incorrectly extracted dot bullets to Markdown bullets.
    text = re.sub(
        r"(?m)^[ \t]*\.[ \t]+(?=[A-Za-z])",
        "- ",
        text,
    )

    # Normalize heading levels for known roles.
    roles = [
        "Chief Executive Officer",
        "Chief Operating Officer",
        "Chief Financial Officer",
        "Chief Technology Officer",
        "Vice President of Sales",
        "Vice President of Marketing",
        "Vice President of Operations",
        "Vice President of Human Resources",
        "Vice President of Research and Development",
        "Vice President of Product Management",
        "Director of Sales",
        "Director of Marketing",
        "Director of Operations",
        "Director of Human Resources",
        "Director of Research and Development",
        "Director of Product Management",
        "Senior Manager of Sales",
        "Senior Manager of Marketing",
        "Senior Manager of Operations",
        "Senior Manager of Human Resources",
        "Senior Manager of Research and Development",
        "Senior Manager of Product Management",
        "Manager of Sales",
        "Manager of Marketing",
        "Manager of Operations",
        "Manager of Human Resources",
        "Manager of Research and Development",
        "Manager of Product Management",
        "Sales Representative",
        "Customer Service Representative",
    ]

    role_names = set(roles)
    lines = []

    for line in text.splitlines():
        stripped = line.strip()
        heading_text = re.sub(r"^#{1,6}\s*", "", stripped)

        if heading_text in role_names:
            lines.append(f"## {heading_text}")
        else:
            lines.append(line.rstrip())

    text = "\n".join(lines)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join words broken across PDF lines.
    text = re.sub(
        r"(?<=[A-Za-z])-\n(?=[A-Za-z])",
        "-",
        text,
    )

    # Join wrapped lines within paragraphs and bullet items.
    text = re.sub(
        r"(?<=[^\n])\n(?=[a-z])",
        " ",
        text,
    )

    return text.strip() + "\n"


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    original = INPUT_PATH.read_text(encoding="utf-8")
    cleaned = clean_markdown(original)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(cleaned, encoding="utf-8")

    print("Markdown cleanup completed!")
    print(f"Original characters: {len(original)}")
    print(f"Cleaned characters: {len(cleaned)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
