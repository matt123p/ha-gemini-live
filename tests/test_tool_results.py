"""Compatibility with HA Core 2026.10 ToolResult and earlier JSON results."""
from dataclasses import dataclass
from datetime import date, time

from google.genai import types
import pytest
from homeassistant.helpers import llm
from gemini_live.stt import _validate_tool_results


@dataclass(slots=True)
class ToolResult:
    data: dict
    error: bool = False


@pytest.fixture
def tool_result_type(monkeypatch):
    # Model Core's new contract when the local HA fixture predates 2026.10.
    monkeypatch.setattr(llm, "ToolResult", ToolResult, raising=False)
    return ToolResult


def test_core_tool_result_is_accepted_by_gemini(tool_result_type):
    result = tool_result_type({"result": "Live context"})
    payload = _validate_tool_results(result)
    response = types.FunctionResponse(name="GetLiveContext", response=payload)
    assert response.response == {"result": "Live context"}


def test_core_tool_error_preserves_failure(tool_result_type):
    result = tool_result_type({"message": "Unavailable"}, error=True)
    payload = _validate_tool_results(result)
    response = types.FunctionResponse(name="GetLiveContext", response=payload)
    assert response.response == {"error": {"message": "Unavailable"}}


def test_core_tool_data_still_normalizes_dates(tool_result_type):
    result = tool_result_type({"events": [{"day": date(2026, 10, 9), "time": time(12, 5)}]})
    assert _validate_tool_results(result) == {
        "events": [{"day": "2026-10-09", "time": "12:05:00"}]
    }


def test_plain_dictionary_remains_supported(tool_result_type):
    result = {"success": True, "result": {"day": date(2026, 10, 9)}}
    assert _validate_tool_results(result) == {
        "success": True, "result": {"day": "2026-10-09"}
    }


def test_old_core_without_tool_result_class(monkeypatch):
    monkeypatch.delattr(llm, "ToolResult", raising=False)
    assert _validate_tool_results({"error": "Unavailable"}) == {"error": "Unavailable"}
