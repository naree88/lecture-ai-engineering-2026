"""Offline checks for the question CLI and its multi-turn chat mode."""

from copy import deepcopy
from unittest import mock
import sys

import pytest
from google.genai import errors, models, types

from app import ask as ask_module


def response(text, *, usage=True):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)])
            )
        ],
        usage_metadata=(
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=12, candidates_token_count=5
            )
            if usage
            else None
        ),
    )


@pytest.fixture
def api(monkeypatch):
    """Keep the real SDK chat, but replace its network-producing model call."""
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    calls, outcomes = [], []

    def generate(self, *, model, contents, config=None):
        calls.append({"model": model, "contents": deepcopy(contents)})
        if not outcomes:
            pytest.fail("Unexpected additional model request")
        result = outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(models.Models, "generate_content", generate)
    return calls, outcomes


def feed_input(monkeypatch, lines):
    lines = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))


def turns(call):
    return [
        (content.role, "".join(part.text or "" for part in content.parts or []))
        for content in call["contents"]
    ]


def test_chat_sends_previous_user_and_model_turns(api, monkeypatch, capsys):
    calls, outcomes = api
    outcomes.extend([response("Understood."), response("Cobalt.")])
    feed_input(monkeypatch, ["What was the item?", "/exit"])

    ask_module.chat("The item is cobalt.")

    assert len(calls) == 2
    assert turns(calls[1]) == [
        ("user", "The item is cobalt."),
        ("model", "Understood."),
        ("user", "What was the item?"),
    ]
    assert all(call["model"] == ask_module.MODEL for call in calls)
    output = capsys.readouterr().out
    assert "Understood." in output and "Cobalt." in output
    assert output.count("input tokens: 12") == 2
    assert output.count("output tokens: 5") == 2


def test_chat_recovers_without_recording_failed_turn(api, monkeypatch, capsys):
    calls, outcomes = api
    outcomes.extend(
        [
            response("First answer."),
            errors.APIError(
                429,
                {"error": {"message": "Rate limited", "status": "RESOURCE_EXHAUSTED"}},
            ),
            response("Recovered answer."),
        ]
    )
    feed_input(monkeypatch, ["Failed question", "Follow-up", "/exit"])

    ask_module.chat("First question")

    assert len(calls) == 3
    assert turns(calls[2]) == [
        ("user", "First question"),
        ("model", "First answer."),
        ("user", "Follow-up"),
    ]
    assert "Recovered answer." in capsys.readouterr().out


@pytest.mark.parametrize("command", ["/exit", "/QUIT"])
def test_blank_input_and_exit_do_not_call_model(api, monkeypatch, command):
    calls, _ = api
    feed_input(monkeypatch, ["", "   ", command])

    ask_module.chat()

    assert calls == []


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_chat_ends_cleanly_on_input_interrupt(api, monkeypatch, exception):
    calls, outcomes = api
    outcomes.append(response("Answer."))

    def interrupted(prompt):
        raise exception

    monkeypatch.setattr("builtins.input", interrupted)

    ask_module.chat("Question")

    assert len(calls) == 1


def test_missing_usage_and_text_are_handled(api, monkeypatch, capsys):
    calls, outcomes = api
    outcomes.extend([response("Answer without usage.", usage=False), types.GenerateContentResponse()])
    feed_input(monkeypatch, ["Another question", "/quit"])

    ask_module.chat("Question")

    assert len(calls) == 2
    output = capsys.readouterr().out
    assert "Answer without usage." in output
    assert "[No text response returned.]" in output
    assert "[usage]" not in output


def test_single_question_behavior_and_usage_are_preserved(api, capsys):
    calls, outcomes = api
    outcomes.append(response("Single answer."))

    assert ask_module.ask("Single question") == "Single answer."

    assert calls == [{"model": ask_module.MODEL, "contents": "Single question"}]
    output = capsys.readouterr().out
    assert "input tokens: 12" in output
    assert "output tokens: 5" in output


@pytest.mark.parametrize(
    ("arguments", "chat_mode", "question"),
    [
        ([], False, ask_module.DEFAULT_QUESTION),
        (["Question"], False, "Question"),
        ([""], False, ""),
        (["--chat"], True, None),
        (["--chat", "Initial question"], True, "Initial question"),
    ],
)
def test_main_routes_arguments(monkeypatch, arguments, chat_mode, question):
    single = mock.Mock(return_value="Single answer.")
    conversation = mock.Mock()
    monkeypatch.setattr(ask_module, "ask", single)
    monkeypatch.setattr(ask_module, "chat", conversation)
    monkeypatch.setattr(sys, "argv", ["app.ask", *arguments])

    ask_module.main()

    if chat_mode:
        conversation.assert_called_once_with(question)
        single.assert_not_called()
    else:
        single.assert_called_once_with(question)
        conversation.assert_not_called()
