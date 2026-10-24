"""Lecture 1: Ask one question, or keep a conversation open.

Usage:
    uv run python -m app.ask "What is LoRA?"
    uv run python -m app.ask              # Use the course-specific question.
    uv run python -m app.ask --chat
    uv run python -m app.ask --chat "What is LoRA?"

Chat history is kept in memory until this process ends. Each follow-up includes
the previous questions and model responses; input token counts include history.
This script has no tools or course materials, so course-specific answers can
still be incorrect. Lecture 3 adds glossary tools to address that limitation.
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# The correct answer to this course-specific question is in data/glossary.json.
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _print_usage(response: types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] input tokens: {usage.prompt_token_count}, "
            f"output tokens: {usage.candidates_token_count}"
        )


def ask(question: str) -> str:
    """Ask a single question without retaining conversation history."""
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
    _print_usage(response)
    return response.text or "[No text response returned.]"


def chat(initial_question: str | None = None) -> None:
    """Ask follow-up questions using one SDK-managed conversation."""
    print("Chat mode: type /exit or /quit to finish. History lasts this session.")
    with genai.Client(api_key=get_api_key()) as client:
        conversation = client.chats.create(model=MODEL)
        pending_question = initial_question
        try:
            while True:
                if pending_question is not None:
                    question = pending_question
                    pending_question = None
                    print(f"You: {question}")
                else:
                    question = input("You: ")
                question = question.strip()
                if question.lower() in {"/exit", "/quit"}:
                    break
                if not question:
                    continue

                try:
                    response = conversation.send_message(question)
                except errors.APIError as e:
                    print(api_error_message(e.code, e.message))
                else:
                    _print_usage(response)
                    print(f"Assistant: {response.text or '[No text response returned.]'}\n")
        except (EOFError, KeyboardInterrupt):
            print()
    print("Chat ended.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask an LLM a question or start a chat.")
    parser.add_argument("question", nargs="?", help="Optional first question.")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Keep conversation history for interactive follow-up questions.",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat(args.question)
        else:
            question = args.question if args.question is not None else DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
