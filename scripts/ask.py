import argparse
import json
import sys
from dataclasses import asdict

from graphrag.chatbot.agent import BaselineAgent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the Zava Foundry baseline agent a question."
    )
    parser.add_argument("question", help="Question in quotes.")
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output the answer, sources, and tool-call count as JSON.",
    )
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("question must not be empty")

    try:
        result = BaselineAgent().ask(args.question)
    except (RuntimeError, ValueError) as error:
        print(f"Cannot answer: {error}", file=sys.stderr)
        return 1
    except Exception:
        print(
            "Request failed. Check Azure sign-in, Foundry/Neo4j settings, "
            "permissions, and vector-index readiness. Credentials were not printed.",
            file=sys.stderr,
        )
        return 1

    if args.as_json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    else:
        print(result.answer)
        if result.sources:
            print("\nSources:")
            for source in result.sources:
                pages = ", ".join(str(page) for page in source.get("pages", [])) or "unknown"
                print(
                    f"[{source['citation_id']}] {source['source_document']}; "
                    f"pages {pages}; chunk {source['chunk_id']}"
                )
        print(f"\nSearch tool calls: {result.tool_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())