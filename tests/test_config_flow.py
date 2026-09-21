"""Tests for the config flow schema's barge-in and affective dialog settings."""

from typing import Any

import voluptuous as vol
from gemini_live.config_flow import (
    _barge_in_errors,
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
    gemini_keys = [
        marker.schema for marker in _provider_schema(PROVIDER_GEMINI).schema
    ]
    openai_keys = [
        marker.schema for marker in _provider_schema(PROVIDER_OPENAI).schema
    ]

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
    assert _needs_model_specific_refresh(extended)

    extended[CONF_AFFECTIVE_DIALOG] = True
    assert not _needs_model_specific_refresh(extended)


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


def test_schema_shows_barge_in_for_both_providers() -> None:
    for provider in (PROVIDER_GEMINI, PROVIDER_OPENAI):
        schema = _provider_schema(provider)
        keys = [marker.schema for marker in schema.schema]
        assert CONF_SUPPORT_BARGE_IN in keys


def test_barge_in_defaults_to_false() -> None:
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


def test_barge_in_setting_persists_through_reconfigure() -> None:
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


def test_barge_in_requires_core_interruption_support(
    monkeypatch,
) -> None:
    """Reject enabling barge-in unless Core has the complete interrupt path."""
    monkeypatch.setattr(
        "gemini_live.config_flow.supports_tts_interruption", lambda: False
    )

    assert _barge_in_errors({CONF_SUPPORT_BARGE_IN: True}) == {
        "base": "barge_in_unsupported"
    }
    assert _barge_in_errors({CONF_SUPPORT_BARGE_IN: False}) == {}


def test_affective_dialog_only_shown_for_supported_models() -> None:
    for config, expected in (
        ({CONF_MODEL: "gemini-3.8-live"}, True),
        ({CONF_MODEL: "gemini-3.8-live-extended-thinking"}, True),
        ({CONF_MODEL: "gemini-3.1-flash-live-preview"}, False),
        ({}, False),
    ):
        schema = _provider_schema(PROVIDER_GEMINI, config)
        keys = [marker.schema for marker in schema.schema]
        assert (CONF_AFFECTIVE_DIALOG in keys) is expected

    schema = _provider_schema(
        PROVIDER_OPENAI, {CONF_MODEL: "gpt-realtime-2.1"}
    )
    keys = [marker.schema for marker in schema.schema]
    assert CONF_AFFECTIVE_DIALOG not in keys


def test_affective_dialog_defaults_to_false() -> None:
    schema = _provider_schema(PROVIDER_GEMINI, {CONF_MODEL: "gemini-3.8-live"})

    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gemini-3.8-live",
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )

    assert result[CONF_AFFECTIVE_DIALOG] is DEFAULT_AFFECTIVE_DIALOG
    assert DEFAULT_AFFECTIVE_DIALOG is False


def test_affective_dialog_setting_persists_through_reconfigure() -> None:
    schema = _provider_schema(
        PROVIDER_GEMINI,
        {CONF_MODEL: "gemini-3.8-live", CONF_AFFECTIVE_DIALOG: True},
    )

    result = schema(
        {
            CONF_API_KEY: "key",
            CONF_MODEL: "gemini-3.8-live",
            CONF_VOICE: _VALID_VOICE[PROVIDER_GEMINI],
        }
    )

    assert result[CONF_AFFECTIVE_DIALOG] is True
