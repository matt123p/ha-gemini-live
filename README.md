# Gemini Live and GPT Realtime for Home Assistant

[![HACS validation](https://img.shields.io/github/actions/workflow/status/matt123p/ha-gemini-live/validate.yml?branch=main&label=HACS%20validation)](https://github.com/matt123p/ha-gemini-live/actions/workflows/validate.yml)
[![GitHub release](https://img.shields.io/github/v/release/matt123p/ha-gemini-live)](https://github.com/matt123p/ha-gemini-live/releases)
[![License](https://img.shields.io/github/license/matt123p/ha-gemini-live)](LICENSE)

This custom Home Assistant integration connects the Home Assistant voice pipeline
directly to either Google's Gemini Live API or OpenAI's Realtime API.

Doing this has the advantage of reducing the time it takes to reply because the 
Speech-to-text and the Text-to-speech are done natively by the Live model.

It streams microphone audio to the selected provider, lets the model call Home
Assistant's exposed Assist tools, and plays the native spoken response back through
the pipeline. This bypasses separate Speech-To-Text (STT) and Text-to-Speech (TTS)
cloud calls.

**NOTE:** The model transcribes the user's speech, but response transcription is
optional. Enabling it can delay the start of playback in Home Assistant.

> [!IMPORTANT]
> This is an independent community integration. It is not the official Home
> Assistant Google Gemini integration and is not affiliated with Google, Google
> DeepMind, OpenAI, or the Open Home Foundation.


## How It Works

A normal Assist pipeline has three separate stages:

1. Speech-to-text turns microphone audio into text.
2. A conversation agent handles the text and returns response text.
3. Text-to-speech synthesizes that response for playback.

The integration deliberately bends that arrangement:

1. The selected live STT entity doesn't perform Speech-to-text but instead streams
   the microphone audio to Gemini or OpenAI.
2. In the same live turn, the model transcribes the user, calls any required Home
   Assistant Assist tools, and starts producing native audio.
3. The conversation entity passes a provider-specific, per-turn placeholder through
   Home Assistant. The unique ID prevents Home Assistant's
   persistent TTS cache from replaying audio from an earlier turn.
4. The matching TTS entity doesn't actually perform Text-to-speech but instead
   streams the native audio directly from the live model.

This design avoids sending the request through separate STT, LLM, and TTS cloud
calls.

Live connections are kept open per Home Assistant conversation ID.
Follow-up turns in the same conversation reuse the existing Live session, while
a new conversation receives a separate session. Home Assistant expires inactive
conversation sessions after its configured chat-session timeout, at which point
the matching Live connection is closed.

The shared pipeline talks to a provider-neutral live session contract. Google
Gemini SDK details live in `gemini.py`, while OpenAI Realtime WebSocket details
live in `openai.py`; neither provider's event or configuration format leaks into
the shared STT, conversation, or session-manager code.

## Difference From The Official Gemini Integration

Home Assistant includes an official
[Google Gemini integration](https://www.home-assistant.io/integrations/google_generative_ai_conversation/).
Gemini Live can be used side-by-side with the official integration.

| Capability | Gemini Live, this repository | Official Google Gemini integration |
| --- | --- | --- |
| Primary goal | Native, end-to-end Gemini Live voice turns | General Gemini conversation and content generation |
| Microphone audio | Streamed directly into a Gemini Live session | Uses the normal Assist pipeline before the conversation agent |
| Spoken reply | Native audio returned by the same Live session | Provides a standalone Google Gemini TTS entity |
| Home control | Calls the Home Assistant Assist LLM API tools | Can control Home Assistant through configured LLM APIs |
| Barge-in | Experimental; supported with the required Home Assistant Core interrupt path and a compatible full-duplex satellite | Not provided by the integration |
| Typed conversation | Supported (but not recommended - use the offical Gemini integration instead) | Supported |
| Standalone TTS | Not supported - Use the offical Gemini integration instead | Supported with `tts.speak`, including voice options |
| Image/PDF analysis | Not supported - Use the offical Gemini integration instead | Supported by the `generate_content` action |
| Google Search option | Native Live API Search grounding, enabled per entry | Google Search support for its configured conversation models |
| Model stability | Includes stable Gemini 3.8 Live models and legacy preview models | Offers the models and settings supported by Home Assistant Core |
| Support channel | Community repository issues | Home Assistant Core issue tracker and documentation |

### Recommended: Set Up Both Integrations

It is recommended to set up the official Google Gemini integration first to
confirm that its conversation agent, Home Assistant control, and API key all
work correctly. This provides a known-good baseline before adding Gemini Live.

You *can* have two completely separate voice assistants in Home Assistant, so
once you have installed the official Gemini integration, you do not have to 
uninstall it to install the Gemini Live integration.

Use the standard Gemini assistant for typed conversations. Although Gemini Live
supports typed and text-only operation, it uses a Live audio-capable model to 
do so. That is generally more expensive than using a model intended for 
text-only requests. Use the Gemini Live assistant when you specifically want 
direct audio streaming and native spoken responses.

It is also a bit easier to debug the integration of Home Assistant with a the
voice assistant in the offical integration.  If you are having issues with Gemini
refusing to read or control something in Home Assistant, go back to the offical
integration and see if it works there first.

## Prerequisites

Before installing, you need:

- A working Home Assistant installation with Assist pipeline support.
- [HACS](https://www.hacs.xyz/) for the recommended installation method.
- A Google Gemini API key created in
  [Google AI Studio](https://aistudio.google.com/app/apikey).
- Gemini Live API access in the region and account associated with the key; or
  an OpenAI API key with Realtime API access from the
  [OpenAI platform](https://platform.openai.com/api-keys).
- An Assist-capable voice device, browser, or companion app if you want to use
  voice input and playback.
- Entities and scripts exposed to Assist if you want the model to control them.

For current language support, see the provider documentation for
[Gemini Live](https://ai.google.dev/gemini-api/docs/live-api/capabilities#supported-languages)
and [OpenAI audio](https://developers.openai.com/api/docs/guides/text-to-speech#supported-languages).

Review Google's current
[Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing),
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), and data-use
terms before sending household audio or entity information.

## Installation

### HACS

Add this repository as a custom repository in HACS:

1. Open HACS in Home Assistant.
2. Select the three-dot menu, then **Custom repositories**.
3. Enter `https://github.com/matt123p/ha-gemini-live`.
4. Select **Integration** as the category and add the repository.
5. Find **Gemini Live** in HACS and select **Download**.
6. Restart Home Assistant when HACS asks you to.

[![Open your Home Assistant instance and open this repository in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=matt123p&repository=ha-gemini-live&category=integration)

### Manual

1. Download the latest release from GitHub.
2. Copy `custom_components/gemini_live` into your Home Assistant configuration
   directory at `custom_components/gemini_live`.
3. Restart Home Assistant.

The final path must contain:

```text
<config>/
└── custom_components/
    └── gemini_live/
        ├── __init__.py
        ├── manifest.json
        ├── stt.py
        ├── conversation.py
        └── tts.py
```

### Reduce ESPHome Assist Satellite Latency

For the lowest response latency on ESPHome Assist satellites, you also need the
changes from [home-assistant/core#173712](https://github.com/home-assistant/core/pull/173712).
Without this change, Home Assistant buffers the complete TTS audio stream before
it starts sending audio to the satellite.

**Good News** - this PR is now in the latest version of Home Assistant.  Just make sure you
upgrade to the latest version.

## Configure The Integration

1. In Home Assistant, open **Settings > Devices & services**.
2. Select **Add integration**.
3. Search for **Gemini Live**.
4. Select Google Gemini or OpenAI as the provider.
5. Enter that provider's API key and select a Realtime model and voice.
6. Optionally enter a system instruction.
7. Leave detailed logging disabled unless you are diagnosing a problem.
8. Select **Submit**.

### Configuration Options

| Option | Description |
| --- | --- |
| Provider | Google Gemini Live or OpenAI GPT Realtime. Existing entries default to Gemini. |
| API key | API key for the selected provider, used for every live connection. |
| Live model | Provider-specific Realtime model used for voice and typed conversations. |
| Voice | Provider-specific built-in voice used for native audio responses. |
| System instruction | Optional personality and behavior instruction. Instructions from the selected Home Assistant LLM APIs are appended automatically. |
| LLM APIs | Home Assistant APIs whose tools the model may call. Keep **Assist** selected for smart-home control; integrations such as AI Memory add their own choices, such as **Memory Management**. Multiple APIs can be enabled together. |
| Detailed logging | Enables verbose logs from this custom integration. These logs can contain transcripts, model details, and tool-call information. |
| Transcribe Gemini / GPT | Streams the model's spoken-response transcript into Home Assistant while native audio is still arriving. Disabled by default for the lowest playback latency. |
| Google Search grounding | Gemini only. Gives the Live model access to Google's built-in Search tool for current or verifiable web information. Optional and disabled by default. Search use may add Gemini API charges. |
| Thinking level | Gemini 3.8 Live Extended Thinking only. Choose low, medium, or high background reasoning. Higher levels may improve complex answers while increasing latency and cost. |
| Encourage web search | OpenAI only. Encourages the model to use an exposed search-like Assist tool. This is a prompt hint, not Gemini Search grounding. Disabled by default. |
| Support barge-in | Experimental: keeps microphone audio streaming while the live model is speaking so the user can interrupt a response. Requires the Home Assistant Core interrupt path and a compatible full-duplex voice client or satellite. Disabled by default. |

For most assistants, choose **Gemini 3.8 Live**: it is optimized for immediate,
low-latency conversation and direct smart-home commands. Choose **Gemini 3.8
Live Extended Thinking** for complex planning, multi-step analysis, or slower
tool workflows. Extended Thinking reasons in the background and may speak brief
progress updates before its final answer, so it can take longer to finish a
request. After selecting it, submit the form once to reveal its **Thinking
level** choice (low, medium, or high). The integration keeps the Home Assistant
turn open until Gemini reports that the full interaction is idle, rather than
treating an intermediate spoken update as the final reply. Settings that do not
apply to the selected model are omitted from the follow-up form.

> [!WARNING]
> Barge-in support is experimental. A standard Assist satellite will not work:
> it must continue sending microphone audio during speaker playback and respond
> to Home Assistant's playback-interrupt signal. Acoustic echo cancellation
> must support double-talk so the assistant does not hear and interrupt itself.
> For an ESPHome implementation, see
> [esphome-aec](https://github.com/matt123p/esphome-aec). See the
> [remote satellite barge-in guide](REMOTE_SATELLITE_BARGE_IN.md) for the full
> Home Assistant Core and satellite requirements.

To change the options later, open **Settings > Devices & services**, select
**Gemini Live**, and select **Configure** or **Reconfigure**.

## Create An Assist Pipeline

Each config entry creates three matching provider entities:

- A speech-to-text bridge.
- A conversation agent.
- A text-to-speech bridge.

Use all three together in the same pipeline:

1. Open **Settings > Voice assistants**.
2. Create a new assistant, or edit a dedicated experimental assistant.
3. Set **Conversation agent** to the configured **Gemini Live** or **GPT Realtime** entity.
4. Turn **off** Prefer handling commands locally.
5. Select the matching provider entry for **Speech-to-text**.
6. Select the matching provider entry for **Text-to-speech**.
7. Save the assistant and test it from the Assist dialog before assigning it to
   voice hardware.

Although mixing the Gemini Live conversation agent with other Speech-to-text or Text-to-speech agents might
just work, it isn't a good idea because the point of using Gemini live is to get the native voice from Gemini
directly.

## Give The Model Access To Home Assistant

Keep **Assist** selected under **LLM APIs** to let the model control or inspect
what Home Assistant exposes through Assist. Keep the exposed set as small as
practical.

1. Open **Settings > Voice assistants**.
2. Open the **Expose** tab.
3. Expose only the entities and scripts Gemini should be allowed to use.
4. Give entities clear names, aliases, areas, and descriptions.
5. Test state queries before testing control commands.

Examples:

- "Turn off the downstairs lights."
- "What is the temperature in the nursery?"
- "Run the good night script."
- "Set the office thermostat to 20 degrees."

Tool execution is performed by Home Assistant through the selected LLM APIs.
The integration sends their tool definitions and results to the live provider
so it can decide what to call and describe the outcome. Third-party integrations
that register an LLM API, including AI Memory, appear in the same selector and
can be enabled alongside Assist.

## Enable Google Search

Google's Live API now supports its built-in
[Google Search grounding tool](https://ai.google.dev/gemini-api/docs/live-api/tools)
in the same session as function-calling tools. This integration can therefore
give Gemini current web information while still exposing Home Assistant's
Assist tools; a second conversation agent, search script, and the official
integration's older workaround are not required.

Enable **Google Search grounding** in the Gemini Live integration options. It is
off by default. When enabled, Gemini decides when a request benefits from Search
and Google executes the search server-side. The integration also instructs
Gemini to use Search for current, changing, time-sensitive, or explicitly
requested online information instead of relying on its training data. This is
different from exposing a Home Assistant search script: Search grounding is a
native Gemini tool, needs no entity exposure, and grounds the answer directly
in web results. Search queries may be billed separately under the Gemini API
pricing for grounding.

All Gemini Live choices offered by this integration support Search grounding:

- Gemini 3.8 Live
- Gemini 3.8 Live Extended Thinking
- Gemini 3.1 Flash Live Preview
- Gemini 2.5 Flash Native Audio Preview (12-2025)

For the lowest latency and to avoid search charges, leave the option disabled
when the assistant only needs Home Assistant state and control. Enable it when
you want current events, changing facts, or web verification.

## Privacy And Security

- This is a cloud integration. Audio and text leave your Home Assistant
  instance and are sent to the provider selected for that entry.
- The provider API key is stored in the Home Assistant config entry.
- Home Assistant entity names, tool schemas, and tool results may be sent to
  Google when Assist control is available.
- Detailed logs can contain transcripts and tool-call details. Disable detailed
  logging after troubleshooting and inspect logs before sharing them.
- Treat prompts and model output as untrusted. Expose only the entities and
  scripts the assistant genuinely needs.
- Do not expose dangerous or irreversible actions without additional
  safeguards.

## Troubleshooting

### Gemini Live Does Not Appear

Confirm that the directory is exactly
`<config>/custom_components/gemini_live`, restart Home Assistant, and clear the
browser cache. Check **Settings > System > Logs** for manifest or dependency
installation errors.

### The Pipeline Returns Silence

Confirm that Gemini Live is selected for all three pipeline stages. The TTS
entity only plays audio cached by the Gemini Live STT/conversation turn. Also
check that your account can access the selected Live model.

### Gemini Answers But Cannot Control Devices

Confirm that the entities are exposed to Assist, have clear names, and can be
controlled by the built-in Assist agent. Gemini cannot call entities or scripts
that Home Assistant does not expose through its Assist LLM API.

### Authentication, Quota, Or Model Errors

Create or verify the API key in Google AI Studio, check the selected project's
quota and billing status, and confirm that the configured preview model is
still available to that account.

### Detailed Logging

Enable **Detailed logging** in the integration options, reproduce one request,
then inspect **Settings > System > Logs**. Disable the option afterward.

## Removing

1. Open **Settings > Devices & services**.
2. Open **Gemini Live** and delete its config entry.
3. Remove the integration in HACS, or manually delete
   `custom_components/gemini_live`.
4. Restart Home Assistant.

## Development

Regenerate the brand icon with:

```bash
python3 scripts/generate_icon.py
```

The generator uses only the Python standard library.

## License

Gemini Live for Home Assistant is available under the [MIT License](LICENSE).
