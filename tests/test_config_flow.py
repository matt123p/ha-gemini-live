"""Tests for the config flow schema's barge-in setting."""

from typing import Any

import voluptuous as vol
from gemini_live.config_flow import _provider_schema
from gemini_live.const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_SUPPORT_BARGE_IN,
    CONF_VOICE,
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
    return list(model_validator.container)


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
