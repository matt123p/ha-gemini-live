"""Config flow for live voice-model providers."""

from collections.abc import Callable
import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm, selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .compat import supports_tts_interruption
from .const import (
    AVAILABLE_MODELS,
    AVAILABLE_VOICES_INFO,
    CONF_AFFECTIVE_DIALOG,
    CONF_API_KEY,
    CONF_DETAILED_LOGGING,
    CONF_ELEVENLABS_AGENT_ID,
    CONF_ENCOURAGE_WEB_SEARCH,
    CONF_ENABLE_SOUND_RESPONSES,
    CONF_MODEL,
    CONF_PROVIDER,
    CONF_RESPONSE_TIMEOUT,
    CONF_SEARCH_GROUNDING,
    CONF_SUPPORT_BARGE_IN,
    CONF_SYSTEM_INSTRUCTION,
    CONF_THINKING_LEVEL,
    CONF_TRANSCRIBE_GEMINI,
    CONF_TRANSCRIBE_GPT,
    CONF_VOICE,
    DEFAULT_AFFECTIVE_DIALOG,
    DEFAULT_ENCOURAGE_WEB_SEARCH,
    DEFAULT_ENABLE_SOUND_RESPONSES,
    DEFAULT_MODEL,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SEARCH_GROUNDING,
    DEFAULT_SUPPORT_BARGE_IN,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_TRANSCRIBE_GEMINI,
    DEFAULT_TRANSCRIBE_GPT,
    DEFAULT_VOICE,
    DOMAIN,
    OPENAI_AVAILABLE_MODELS,
    OPENAI_AVAILABLE_VOICES_INFO,
    OPENAI_DEFAULT_MODEL,
    OPENAI_DEFAULT_VOICE,
    PERSONAPLEX_AVAILABLE_MODELS,
    PERSONAPLEX_AVAILABLE_VOICES_INFO,
    PERSONAPLEX_DEFAULT_MODEL,
    PERSONAPLEX_DEFAULT_VOICE,
    MAX_RESPONSE_TIMEOUT,
    MIN_RESPONSE_TIMEOUT,
    PROVIDER_QWEN,
    PROVIDER_ELEVENLABS,
    CONF_TRANSCRIBE_ELEVENLABS,
    ELEVENLABS_AVAILABLE_MODELS,
    ELEVENLABS_DEFAULT_MODEL,
    ELEVENLABS_DEFAULT_VOICE,
    QWEN_AVAILABLE_MODELS,
    QWEN_DEFAULT_MODEL,
    CONF_TRANSCRIBE_QWEN,
    QWEN_AVAILABLE_VOICES,
    QWEN_DEFAULT_VOICE,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_PERSONAPLEX,
    THINKING_LEVELS,
    supports_affective_dialog,
    supports_thinking_level,
)
from .elevenlabs import (
    ElevenLabsAgentsError,
    agent_language,
    async_create_agent,
    async_list_voices,
    async_update_agent,
)

_LOGGER = logging.getLogger(__name__)

PROVIDER_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=PROVIDER_GEMINI, label="Google Gemini"),
            selector.SelectOptionDict(value=PROVIDER_OPENAI, label="OpenAI"),
            selector.SelectOptionDict(value=PROVIDER_QWEN, label="QwenCloud"),
            selector.SelectOptionDict(
                value=PROVIDER_ELEVENLABS, label="ElevenLabs Agents"
            ),
            selector.SelectOptionDict(
                value=PROVIDER_PERSONAPLEX, label="fal.ai PersonaPlex"
            ),
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)


def _model_selector(
    models: list[str], *, custom_value: bool = False
) -> selector.SelectSelector:
    """Return a compact dropdown for a provider's model choices."""
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=models,
            mode=selector.SelectSelectorMode.DROPDOWN,
            custom_value=custom_value,
        )
    )


def _elevenlabs_voice_selector(
    voices: list[tuple[str, str]],
) -> selector.SelectSelector:
    """Return the account's ElevenLabs voices, allowing any voice ID."""
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=[
                selector.SelectOptionDict(value=voice_id, label=name)
                for voice_id, name in voices
            ],
            mode=selector.SelectSelectorMode.DROPDOWN,
            custom_value=True,
        )
    )


async def _async_elevenlabs_voices(
    hass: HomeAssistant, api_key: str
) -> list[tuple[str, str]] | None:
    """Return the account's voices, or None when the key is not accepted."""
    try:
        return await async_list_voices(async_get_clientsession(hass), api_key)
    except ElevenLabsAgentsError:
        return None


async def _async_elevenlabs_agent(
    hass: HomeAssistant, api_key: str, agent_id: str | None, title: str
) -> str:
    """Repair the integration's agent, or create it when it is missing."""
    http_session = async_get_clientsession(hass)
    language = agent_language(hass.config.language)
    if agent_id:
        try:
            await async_update_agent(http_session, api_key, agent_id, language)
        except ElevenLabsAgentsError as exc:
            # Only a missing agent gets replaced: deleted in the dashboard, or
            # the API key now names another account. Other errors must not
            # leave a second agent behind.
            if exc.status != 404:
                raise
        else:
            return agent_id
    return await async_create_agent(
        http_session, api_key, f"Home Assistant ({title})", language
    )


GEMINI_VOICE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(
                value=name,
                label=f"{name} - {gender}, {description}",
            )
            for name, gender, description in AVAILABLE_VOICES_INFO
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)

OPENAI_VOICE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=name, label=f"{name} - {description}")
            for name, description in OPENAI_AVAILABLE_VOICES_INFO
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)

PERSONAPLEX_VOICE_SELECTOR = selector.SelectSelector(
    selector.SelectSelectorConfig(
        options=[
            selector.SelectOptionDict(value=name, label=f"{name} - {description}")
            for name, description in PERSONAPLEX_AVAILABLE_VOICES_INFO
        ],
        mode=selector.SelectSelectorMode.DROPDOWN,
    )
)


def _provider(config: dict[str, Any]) -> str:
    """Return the configured provider, defaulting legacy entries to Gemini."""
    return config.get(CONF_PROVIDER, PROVIDER_GEMINI)


def _provider_schema(
    provider: str,
    config: dict[str, Any] | None = None,
    hass: HomeAssistant | None = None,
    voices: list[tuple[str, str]] | None = None,
) -> vol.Schema:
    """Build a provider-specific setup/options schema.

    ElevenLabs voices are account-specific, so its voice field is only shown
    once ``voices`` were fetched with the entered API key.
    """
    current = config or {}
    is_elevenlabs = provider == PROVIDER_ELEVENLABS
    is_qwen = provider == PROVIDER_QWEN
    is_openai = provider == PROVIDER_OPENAI
    is_personaplex = provider == PROVIDER_PERSONAPLEX
    models = (
        QWEN_AVAILABLE_MODELS
        if is_qwen
        else PERSONAPLEX_AVAILABLE_MODELS
        if is_personaplex
        else OPENAI_AVAILABLE_MODELS
        if is_openai
        else AVAILABLE_MODELS
    )
    default_model = (
        QWEN_DEFAULT_MODEL
        if is_qwen
        else PERSONAPLEX_DEFAULT_MODEL
        if is_personaplex
        else OPENAI_DEFAULT_MODEL
        if is_openai
        else DEFAULT_MODEL
    )
    default_voice = (
        PERSONAPLEX_DEFAULT_VOICE
        if is_personaplex
        else OPENAI_DEFAULT_VOICE
        if is_openai
        else DEFAULT_VOICE
    )
    voice_selector = (
        PERSONAPLEX_VOICE_SELECTOR
        if is_personaplex
        else OPENAI_VOICE_SELECTOR
        if is_openai
        else GEMINI_VOICE_SELECTOR
    )
    if is_qwen:
        default_voice = QWEN_DEFAULT_VOICE
        voice_selector = _model_selector(QWEN_AVAILABLE_VOICES)
    if is_elevenlabs:
        models = ELEVENLABS_AVAILABLE_MODELS
        default_model = ELEVENLABS_DEFAULT_MODEL
        default_voice = ELEVENLABS_DEFAULT_VOICE
        voice_selector = _elevenlabs_voice_selector(voices or [])
    transcribe_key = (
        CONF_TRANSCRIBE_ELEVENLABS
        if is_elevenlabs
        else CONF_TRANSCRIBE_QWEN
        if is_qwen
        else CONF_TRANSCRIBE_GPT
        if is_openai
        else CONF_TRANSCRIBE_GEMINI
    )
    default_transcribe = (
        DEFAULT_TRANSCRIBE_GPT if is_openai else DEFAULT_TRANSCRIBE_GEMINI
    )

    api_key_field = (
        vol.Required(CONF_API_KEY, default=current[CONF_API_KEY])
        if CONF_API_KEY in current
        else vol.Required(CONF_API_KEY)
    )
    fields: dict[vol.Marker, Any] = {
        api_key_field: str,
        vol.Required(
            CONF_MODEL,
            default=current.get(CONF_MODEL, default_model),
        ): _model_selector(models, custom_value=is_elevenlabs),
    }
    if not is_elevenlabs or voices is not None:
        fields[
            vol.Required(
                CONF_VOICE,
                default=current.get(CONF_VOICE, default_voice),
            )
        ] = voice_selector
    fields |= {
        vol.Optional(
            CONF_SYSTEM_INSTRUCTION,
            description={"suggested_value": current.get(CONF_SYSTEM_INSTRUCTION, "")},
        ): str,
        vol.Optional(
            CONF_DETAILED_LOGGING,
            default=current.get(CONF_DETAILED_LOGGING, False),
        ): selector.BooleanSelector(),
        vol.Optional(
            transcribe_key,
            default=current.get(transcribe_key, default_transcribe),
        ): selector.BooleanSelector(),
        vol.Optional(
            CONF_RESPONSE_TIMEOUT,
            default=current.get(CONF_RESPONSE_TIMEOUT, DEFAULT_RESPONSE_TIMEOUT),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=MIN_RESPONSE_TIMEOUT,
                max=MAX_RESPONSE_TIMEOUT,
                step=1,
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement="seconds",
            )
        ),
    }
    if not is_personaplex:
        fields[
            vol.Optional(
                CONF_ENABLE_SOUND_RESPONSES,
                default=current.get(
                    CONF_ENABLE_SOUND_RESPONSES, DEFAULT_ENABLE_SOUND_RESPONSES
                ),
            )
        ] = selector.BooleanSelector()
    if hass is not None and not is_personaplex:
        apis = [
            selector.SelectOptionDict(value=api.id, label=api.name)
            for api in llm.async_get_apis(hass)
        ]
        known_api_ids = {api["value"] for api in apis}
        selected_apis = current.get(CONF_LLM_HASS_API, [llm.LLM_API_ASSIST])
        if isinstance(selected_apis, str):
            selected_apis = [selected_apis]
        selected_apis = [api_id for api_id in selected_apis if api_id in known_api_ids]
        fields[
            vol.Optional(
                CONF_LLM_HASS_API,
                default=selected_apis,
                description={"suggested_value": selected_apis},
            )
        ] = selector.SelectSelector(
            selector.SelectSelectorConfig(options=apis, multiple=True)
        )
    if provider == PROVIDER_GEMINI:
        # All Gemini Live models offered by this integration support Google's
        # server-side Search grounding tool. It is a separate, opt-in
        # preference from the former Assist search-tool prompt hint.
        fields[
            vol.Optional(
                CONF_SEARCH_GROUNDING,
                default=current.get(
                    CONF_SEARCH_GROUNDING,
                    DEFAULT_SEARCH_GROUNDING,
                ),
            )
        ] = selector.BooleanSelector()
    elif is_openai:
        fields[
            vol.Optional(
                CONF_ENCOURAGE_WEB_SEARCH,
                default=current.get(
                    CONF_ENCOURAGE_WEB_SEARCH,
                    DEFAULT_ENCOURAGE_WEB_SEARCH,
                ),
            )
        ] = selector.BooleanSelector()
    if not is_personaplex and not is_elevenlabs and supports_tts_interruption():
        # Barge-in needs Core's complete TTS interruption path. When the
        # installed Core cannot propagate interruptions, the setting cannot do
        # anything, so it is hidden entirely and never blocks setup.
        fields[
            vol.Optional(
                CONF_SUPPORT_BARGE_IN,
                default=current.get(
                    CONF_SUPPORT_BARGE_IN,
                    DEFAULT_SUPPORT_BARGE_IN,
                ),
            )
        ] = selector.BooleanSelector()
    # The affective-dialog switch is Gemini-specific and only rendered for
    # models that support it; Home Assistant config forms cannot render a
    # disabled field, so it is hidden instead.
    if (
        provider == PROVIDER_GEMINI
        and supports_affective_dialog(current.get(CONF_MODEL))
    ):
        fields[
            vol.Optional(
                CONF_AFFECTIVE_DIALOG,
                default=current.get(
                    CONF_AFFECTIVE_DIALOG,
                    DEFAULT_AFFECTIVE_DIALOG,
                ),
            )
        ] = selector.BooleanSelector()
    if (
        provider == PROVIDER_GEMINI
        and supports_thinking_level(current.get(CONF_MODEL))
    ):
        fields[
            vol.Optional(
                CONF_THINKING_LEVEL,
                default=current.get(
                    CONF_THINKING_LEVEL,
                    DEFAULT_THINKING_LEVEL,
                ),
            )
        ] = _model_selector(THINKING_LEVELS)
    return vol.Schema(fields)


def _strip_unsupported_settings(user_input: dict[str, Any]) -> dict[str, Any]:
    """Drop settings the selected model or Core does not support."""
    if not supports_affective_dialog(user_input.get(CONF_MODEL)):
        user_input.pop(CONF_AFFECTIVE_DIALOG, None)
    if not supports_thinking_level(user_input.get(CONF_MODEL)):
        user_input.pop(CONF_THINKING_LEVEL, None)
    if not supports_tts_interruption():
        user_input.pop(CONF_SUPPORT_BARGE_IN, None)
    if user_input.get(CONF_MODEL) in QWEN_AVAILABLE_MODELS:
        user_input.pop(CONF_SEARCH_GROUNDING, None)
        user_input.pop(CONF_ENCOURAGE_WEB_SEARCH, None)
    return user_input


def _needs_model_specific_refresh(user_input: dict[str, Any]) -> bool:
    """Return whether a newly selected model needs its settings form shown."""
    model = user_input.get(CONF_MODEL)
    return (
        supports_thinking_level(model) and CONF_THINKING_LEVEL not in user_input
    ) or (supports_affective_dialog(model) and CONF_AFFECTIVE_DIALOG not in user_input)


async def _async_elevenlabs_step(
    flow: config_entries.ConfigFlow | config_entries.OptionsFlow,
    step_id: str,
    user_input: dict[str, Any] | None,
    current: dict[str, Any],
    title: str,
    finish: Callable[[dict[str, Any]], Any],
):
    """Run one ElevenLabs settings form: check the key, pick a voice, save.

    The first submission without a voice validates the API key and shows the
    form again with the account's voices. The final submission creates or
    repairs the dedicated agent before ``finish`` stores the settings.
    """
    hass = flow.hass
    errors: dict[str, str] = {}
    if user_input is None:
        voices = None
        if api_key := current.get(CONF_API_KEY):
            voices = await _async_elevenlabs_voices(hass, api_key)
        return _elevenlabs_form(flow, step_id, current, voices, errors)
    voices = await _async_elevenlabs_voices(hass, user_input[CONF_API_KEY])
    if voices is None:
        errors[CONF_API_KEY] = "invalid_auth"
    elif CONF_VOICE in user_input:
        try:
            agent_id = await _async_elevenlabs_agent(
                hass,
                user_input[CONF_API_KEY],
                current.get(CONF_ELEVENLABS_AGENT_ID),
                title,
            )
        except ElevenLabsAgentsError as exc:
            _LOGGER.error("Could not set up the ElevenLabs agent: %s", exc)
            errors["base"] = "cannot_connect"
        else:
            data = _strip_unsupported_settings(user_input)
            data[CONF_PROVIDER] = PROVIDER_ELEVENLABS
            data[CONF_ELEVENLABS_AGENT_ID] = agent_id
            data.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            return finish(data)
    return _elevenlabs_form(flow, step_id, {**current, **user_input}, voices, errors)


def _elevenlabs_form(
    flow: config_entries.ConfigFlow | config_entries.OptionsFlow,
    step_id: str,
    current: dict[str, Any],
    voices: list[tuple[str, str]] | None,
    errors: dict[str, str],
):
    return flow.async_show_form(
        step_id=step_id,
        data_schema=_provider_schema(PROVIDER_ELEVENLABS, current, flow.hass, voices),
        errors=errors,
        description_placeholders={"provider": "ElevenLabs"},
    )


class GeminiLiveConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configure a realtime voice provider."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Select the live model provider."""
        if user_input is not None:
            return await self.async_step_provider(provider=user_input[CONF_PROVIDER])
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_PROVIDER,
                        default=PROVIDER_GEMINI,
                    ): PROVIDER_SELECTOR
                }
            ),
        )

    async def async_step_provider(
        self,
        user_input=None,
        *,
        provider: str | None = None,
    ):
        """Collect settings for the selected provider."""
        selected_provider = provider or self.context[CONF_PROVIDER]
        self.context[CONF_PROVIDER] = selected_provider
        if selected_provider == PROVIDER_ELEVENLABS:
            return await _async_elevenlabs_step(
                self,
                "provider",
                user_input,
                {},
                "ElevenLabs Agents",
                lambda data: self.async_create_entry(
                    title="ElevenLabs Agents", data=data
                ),
            )
        if user_input is not None:
            if _needs_model_specific_refresh(user_input):
                return self.async_show_form(
                    step_id="provider",
                    data_schema=_provider_schema(
                        selected_provider, user_input, self.hass
                    ),
                    description_placeholders={
                        "provider": "Google Gemini",
                    },
                )
            user_input = _strip_unsupported_settings(user_input)
            user_input[CONF_PROVIDER] = selected_provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            title = {
                PROVIDER_OPENAI: "GPT Realtime",
                PROVIDER_QWEN: "Qwen Realtime",
                PROVIDER_PERSONAPLEX: "PersonaPlex",
            }.get(selected_provider, "Gemini Live")
            return self.async_create_entry(title=title, data=user_input)
        return self.async_show_form(
            step_id="provider",
            data_schema=_provider_schema(selected_provider, hass=self.hass),
            description_placeholders={
                "provider": {
                    PROVIDER_OPENAI: "OpenAI",
                    PROVIDER_QWEN: "QwenCloud",
                    PROVIDER_PERSONAPLEX: "fal.ai PersonaPlex",
                }.get(selected_provider, "Google Gemini")
            },
        )

    async def async_step_reconfigure(self, user_input=None):
        """Reconfigure an existing provider entry."""
        entry = self._get_reconfigure_entry()
        config = {**entry.data, **entry.options}
        provider = _provider(config)
        if provider == PROVIDER_ELEVENLABS:
            return await _async_elevenlabs_step(
                self,
                "reconfigure",
                user_input,
                config,
                entry.title,
                lambda data: self.async_update_reload_and_abort(
                    entry, data_updates=data, options={}
                ),
            )
        if user_input is not None:
            if _needs_model_specific_refresh(user_input):
                return self.async_show_form(
                    step_id="reconfigure",
                    data_schema=_provider_schema(provider, user_input, self.hass),
                )
            user_input = _strip_unsupported_settings(user_input)
            user_input[CONF_PROVIDER] = provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            return self.async_update_reload_and_abort(
                entry,
                data_updates=user_input,
                options={},
            )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_provider_schema(provider, config, self.hass),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Return the options flow."""
        return GeminiLiveOptionsFlowHandler()


class GeminiLiveOptionsFlowHandler(config_entries.OptionsFlow):
    """Manage live provider options."""

    async def async_step_init(self, user_input=None):
        """Update the provider's connection and response settings."""
        config = {**self.config_entry.data, **self.config_entry.options}
        provider = _provider(config)
        if provider == PROVIDER_ELEVENLABS:
            return await _async_elevenlabs_step(
                self,
                "init",
                user_input,
                config,
                self.config_entry.title,
                lambda data: self.async_create_entry(title="", data=data),
            )
        if user_input is not None:
            if _needs_model_specific_refresh(user_input):
                return self.async_show_form(
                    step_id="init",
                    data_schema=_provider_schema(provider, user_input, self.hass),
                )
            user_input = _strip_unsupported_settings(user_input)
            user_input[CONF_PROVIDER] = provider
            user_input.setdefault(CONF_SYSTEM_INSTRUCTION, "")
            return self.async_create_entry(title="", data=user_input)
        return self.async_show_form(
            step_id="init",
            data_schema=_provider_schema(provider, config, self.hass),
        )
