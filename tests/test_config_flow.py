"""Tests for the config flow schema's barge-in and affective dialog settings."""

from types import SimpleNamespace
from typing import Any

import voluptuous as vol
from homeassistant.const import CONF_LLM_HASS_API
from gemini_live.config_flow import (
    _needs_model_specific_refresh,
    _provider_schema,
    _strip_unsupported_settings,
)
from gemini_live.const import (
    CONF_AFFECTIVE_DIALOG,
    CONF_API_KEY,
    CONF_MODEL,
    CONF_SEARCH_GROUNDING,
    CONF_SUPPORT_BARGE_IN,
    CONF_THINKING_LEVEL,
    CONF_VOICE,
    DEFAULT_AFFECTIVE_DIALOG,
    DEFAULT_SEARCH_GROUNDING,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_SUPPORT_BARGE_IN,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_PERSONAPLEX,
)

_VALID_VOICE = {PROVIDER_GEMINI: "Puck", PROVIDER_OPENAI: "marin"}


def _validator(schema: vol.Schema, key: str) -> Any:
    return next(
        value for marker, value in schema.schema.items() if marker.schema == key
    )


def _available_models(schema: vol.Schema) -> list[str]:
    model_validator = _validator(schema, CONF_MODEL)
    if hasattr(model_validator, "container"):
        return list(model_validator.container)
    return list(model_validator.config["options"])


def test_gemini_models_use_dropdown_and_include_extended_thinking() -> None:
    schema = _provider_schema(PROVIDER_GEMINI)
    model_selector = _validator(schema, CONF_MODEL)

    assert model_selector.config["mode"] == "dropdown"
    assert "gemini-3.8-live-extended-thinking" in _available_models(schema)


def test_search_grounding_is_a_gemini_preference() -> None:
    gemini_keys = [marker.schema for marker in _provider_schema(PROVIDER_GEMINI).schema]
    openai_keys = [marker.schema for marker in _provider_schema(PROVIDER_OPENAI).schema]

    assert CONF_SEARCH_GROUNDING in gemini_keys
    assert CONF_SEARCH_GROUNDING not in openai_keys

    result = _provider_schema(PROVIDER_GEMINI)(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gemini-3.8-live",
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )
    assert result[CONF_SEARCH_GROUNDING] is DEFAULT_SEARCH_GROUNDING
    assert DEFAULT_SEARCH_GROUNDING is False


def test_llm_api_selector_lists_registered_apis(monkeypatch) -> None:
    """Offer every registered API, including third-party memory APIs."""
    monkeypatch.setattr(
        "gemini_live.config_flow.llm.async_get_apis",
        lambda _hass: [
            SimpleNamespace(id="assist", name="Assist"),
            SimpleNamespace(id="memory", name="Memory Management"),
        ],
    )

    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_LLM_HASS_API: ["assist", "memory"]},
        SimpleNamespace(),
    )
    api_selector = _validator(schema, CONF_LLM_HASS_API)

    assert api_selector.config["multiple"] is True
    assert api_selector.config["options"] == [
        {"value": "assist", "label": "Assist"},
        {"value": "memory", "label": "Memory Management"},
    ]
    marker = next(
        marker for marker in schema.schema if marker.schema == CONF_LLM_HASS_API
    )
    assert marker.description["suggested_value"] == ["assist", "memory"]


def test_llm_api_selector_defaults_to_assist_and_ignores_missing_apis(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.llm.async_get_apis",
        lambda _hass: [SimpleNamespace(id="assist", name="Assist")],
    )

    schema = _provider_schema(PROVIDER_OPENAI, hass=SimpleNamespace())
    marker = next(
        marker for marker in schema.schema if marker.schema == CONF_LLM_HASS_API
    )
    assert marker.description["suggested_value"] == ["assist"]
    assert marker.default() == ["assist"]

    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gpt-realtime-2.1",
            CONF_VOICE: _VALID_VOICE[PROVIDER_OPENAI],
        }
    )
    assert result[CONF_LLM_HASS_API] == ["assist"]

    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_LLM_HASS_API: ["removed-api"]},
        SimpleNamespace(),
    )
    marker = next(
        marker for marker in schema.schema if marker.schema == CONF_LLM_HASS_API
    )
    assert marker.description["suggested_value"] == []


def test_llm_api_selector_preserves_explicit_empty_selection(monkeypatch) -> None:
    """Do not re-enable Assist after a user explicitly disables all APIs."""
    monkeypatch.setattr(
        "gemini_live.config_flow.llm.async_get_apis",
        lambda _hass: [SimpleNamespace(id="assist", name="Assist")],
    )

    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_LLM_HASS_API: []},
        SimpleNamespace(),
    )
    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gemini-3.8-live",
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )

    assert result[CONF_LLM_HASS_API] == []


def test_llm_api_selector_hidden_for_provider_without_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.llm.async_get_apis",
        lambda _hass: [SimpleNamespace(id="assist", name="Assist")],
    )

    keys = [
        marker.schema
        for marker in _provider_schema(
            PROVIDER_PERSONAPLEX, hass=SimpleNamespace()
        ).schema
    ]
    assert CONF_LLM_HASS_API not in keys


def test_thinking_level_only_shown_for_extended_thinking() -> None:
    for model, expected in (
        ("gemini-3.8-live-extended-thinking", True),
        ("gemini-3.8-live", False),
        ("gemini-3.1-flash-live-preview", False),
    ):
        schema = _provider_schema(PROVIDER_GEMINI, {CONF_MODEL: model})
        keys = [marker.schema for marker in schema.schema]
        assert (CONF_THINKING_LEVEL in keys) is expected

    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_MODEL: "gemini-3.8-live-extended-thinking"},
    )
    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gemini-3.8-live-extended-thinking",
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )
    assert result[CONF_THINKING_LEVEL] == DEFAULT_THINKING_LEVEL


def test_model_change_refreshes_before_model_specific_settings() -> None:
    extended = {CONF_MODEL: "gemini-3.8-live-extended-thinking"}
    assert _needs_model_specific_refresh(extended)

    extended[CONF_THINKING_LEVEL] = "medium"
    assert not _needs_model_specific_refresh(extended)

    native_audio = {CONF_MODEL: "gemini-2.5-flash-native-audio-preview-12-2025"}
    assert _needs_model_specific_refresh(native_audio)

    native_audio[CONF_AFFECTIVE_DIALOG] = True
    assert not _needs_model_specific_refresh(native_audio)


def test_unsupported_model_settings_are_removed() -> None:
    result = _strip_unsupported_settings(
        {
            CONF_MODEL: "gemini-3.1-flash-live-preview",
            CONF_AFFECTIVE_DIALOG: True,
            CONF_THINKING_LEVEL: "high",
        }
    )

    assert CONF_AFFECTIVE_DIALOG not in result
    assert CONF_THINKING_LEVEL not in result


def test_schema_shows_barge_in_for_both_providers(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: True
    )
    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        schema = _provider_schema(provider)
        keys = [marker.schema for marker in schema.schema]
        assert CONF_SUPPORT_BARGE_IN in keys


def test_barge_in_hidden_when_core_lacks_interruption_support(
    monkeypatch,
) -> None:
    """Hide the barge-in option entirely when Core cannot interrupt TTS."""
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: False
    )

    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        schema = _provider_schema(provider)
        keys = [marker.schema for marker in schema.schema]
        assert CONF_SUPPORT_BARGE_IN not in keys

        result = schema(
            {
                CONF_API_KEY: "key",
                CONF_MODEL: _available_models(schema)[0],
                CONF_VOICE: _VALID_VOICE[provider],
            }
        )
        assert CONF_SUPPORT_BARGE_IN not in result


def test_barge_in_setting_stripped_without_core_interruption_support(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: False
    )

    result = _strip_unsupported_settings(
        {
            CONF_MODEL: "gemini-3.8-live",
            CONF_SUPPORT_BARGE_IN: True,
        }
    )

    assert CONF_SUPPORT_BARGE_IN not in result


def test_barge_in_defaults_to_false(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: True
    )
    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        schema = _provider_schema(provider)

        result = schema(
            {
                CONF_API_KEY: "key",
                CONF_MODEL: _available_models(schema)[0],
                CONF_VOICE: _VALID_VOICE[provider],
            }
        )

        assert result[CONF_SUPPORT_BARGE_IN] is DEFAULT_SUPPORT_BARGE_IN
        assert DEFAULT_SUPPORT_BARGE_IN is False


def test_barge_in_setting_persists_through_reconfigure(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: True
    )
    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        schema = _provider_schema(provider, {CONF_SUPPORT_BARGE_IN: True})

        result = schema(
            {
                CONF_API_KEY: "key",
                CONF_MODEL: _available_models(schema)[0],
                CONF_VOICE: _VALID_VOICE[provider],
            }
        )

        assert result[CONF_SUPPORT_BARGE_IN] is True


def test_affective_dialog_only_shown_for_supported_models() -> None:
    for config, expected in (
        ({CONF_MODEL: "gemini-2.5-flash-native-audio-preview-12-2025"}, True),
        ({CONF_MODEL: "gemini-3.8-live"}, False),
        ({CONF_MODEL: "gemini-3.8-live-extended-thinking"}, False),
        ({CONF_MODEL: "gemini-3.1-flash-live-preview"}, False),
        ({}, False),
    ):
        schema = _provider_schema(PROVIDER_GEMINI, config)
        keys = [marker.schema for marker in schema.schema]
        assert (CONF_AFFECTIVE_DIALOG in keys) is expected

    schema = _provider_schema(PROVIDER_OPENAI, {CONF_MODEL: "gpt-realtime-2.1"})
    keys = [marker.schema for marker in schema.schema]
    assert CONF_AFFECTIVE_DIALOG not in keys


def test_affective_dialog_defaults_to_false() -> None:
    model = "gemini-2.5-flash-native-audio-preview-12-2025"
    schema = _provider_schema(PROVIDER_GEMINI, {CONF_MODEL: model})

    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: model,
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )

    assert result[CONF_AFFECTIVE_DIALOG] is DEFAULT_AFFECTIVE_DIALOG
    assert DEFAULT_AFFECTIVE_DIALOG is False


def test_affective_dialog_setting_persists_through_reconfigure() -> None:
    model = "gemini-2.5-flash-native-audio-preview-12-2025"
    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_MODEL: model, CONF_AFFECTIVE_DIALOG: True},
    )

    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: model,
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )

    assert result[CONF_AFFECTIVE_DIALOG] is True
