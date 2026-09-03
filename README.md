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
| Typed conversation | Supported (but not recommended - use the offical Gemini integration instead) | Supported |
| Standalone TTS | Not supported - Use the offical Gemini integration instead | Supported with `tts.speak`, including voice options |
| Image/PDF analysis | Not supported - Use the offical Gemini integration instead | Supported by the `generate_content` action |
| Google Search option | Supported through the official integration's documented search workaround | Supported through the official integration's documented search workaround |
| Model stability | Uses preview Live models | Offers the models and settings supported by Home Assistant Core |
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
| System instruction | Optional personality and behavior instruction. Home Assistant's Assist API prompt is appended automatically. |
| Detailed logging | Enables verbose logs from this custom integration. These logs can contain transcripts, model details, and tool-call information. |
| Transcribe Gemini / GPT | Streams the model's spoken-response transcript into Home Assistant while native audio is still arriving. Disabled by default for the lowest playback latency. |
| Encourage web search | Encourages the model to use an exposed search-like Assist tool for current, recent, time-sensitive, or explicitly requested online information. Disabled by default. |
| Show text | Exposes a callback function so the model can display formatted text/markdown in the Home Assistant chat UI instead of the default placeholder. Only active when response transcription is disabled. Enabled by default. |

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

The selected model can only control or inspect what Home Assistant exposes through Assist.
Keep the exposed set as small as practical.

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

Tool execution is performed by Home Assistant through its Assist LLM API. The
integration sends tool definitions and tool results to Gemini so it can decide
what to call and describe the outcome.

## Enable Google Search

Gemini Live can use the workaround from the official Home Assistant
[Google Gemini integration's Google Search documentation](https://www.home-assistant.io/integrations/google_generative_ai_conversation/#google-search).
The official integration is required because it provides a separate
search-enabled conversation agent for the workaround to call.

This extra agent is necessary because, as the official documentation explains,
the Gemini API does not allow the
[Google Search tool](https://ai.google.dev/gemini-api/docs/google-search) and
function-calling tools such as Home Assistant's Assist tools in the same
request. The workaround exposes a script that sends search queries to the
separate official Gemini agent and returns its answer.

### Set Up The Search Agent

Following the
[official Google Search workaround steps](https://www.home-assistant.io/integrations/google_generative_ai_conversation/#google-search) and check it is working with the offical Gemini integration.

Once exposed, Gemini Live can discover and call `Assist: Search Google` through
Home Assistant's Assist tools.

### Turn on "Encourage web search"
Gemini decides whether to call exposed tools. To make it more likely to use the
search script for current information, enable **Encourage web search** in the
Gemini Live integration options. This adds search-routing instructions to
Gemini's system prompt and strengthens the exposed search tool description. It
does not install, expose, or configure a search tool by itself.

### Use the "Show text" Option for Screen Displays

When **Transcribe Gemini** is turned off (which is recommended for the fastest voice responses), Gemini's spoken reply is normally hidden in the Home Assistant chat UI and only the placeholder `-- gemini live --` is displayed.

If you are using a device with a screen (like a wall tablet, phone, or browser), you can enable the **Show text** option. When enabled, if Gemini decides to give you a detailed list, instructions, links, or code blocks that are better read than listened to, it will display them in the chat UI as formatted text while still speaking to you. If it only has a simple spoken reply, it will continue to show the default placeholder.

## Supported Audio And Languages

The STT entity accepts WAV audio containing 16-bit, 16 kHz, mono PCM. This is
the format used by a compatible Home Assistant Assist pipeline. Gemini's 24 kHz
native response audio is converted to 16 kHz PCM and streamed through the TTS
stage as it arrives. When **Transcribe Gemini** is enabled, Home Assistant also
streams transcript text into TTS and starts playback after its built-in
streaming threshold is reached. Short transcribed replies may therefore wait
until their transcript is complete. Disabling the option starts playback from
the first available audio with only the configured user-transcript wait.

The integration advertises all 78 languages currently listed as supported by
Gemini's native audio models:

| | | | |
| --- | --- | --- | --- |
| Afrikaans (`af`) | Albanian (`sq`) | Amharic (`am`) | Arabic (`ar`) |
| Armenian (`hy`) | Azerbaijani (`az`) | Bangla (`bn`) | Basque (`eu`) |
| Belarusian (`be`) | Bulgarian (`bg`) | Burmese (`my`) | Catalan (`ca`) |
| Cebuano (`ceb`) | Chinese, Mandarin (`cmn`) | Croatian (`hr`) | Czech (`cs`) |
| Danish (`da`) | Dutch (`nl`) | English (`en`) | Estonian (`et`) |
| Filipino (`fil`) | Finnish (`fi`) | French (`fr`) | Galician (`gl`) |
| Georgian (`ka`) | German (`de`) | Greek (`el`) | Gujarati (`gu`) |
| Haitian Creole (`ht`) | Hebrew (`he`) | Hindi (`hi`) | Hungarian (`hu`) |
| Icelandic (`is`) | Indonesian (`id`) | Italian (`it`) | Japanese (`ja`) |
| Javanese (`jv`) | Kannada (`kn`) | Konkani (`kok`) | Korean (`ko`) |
| Lao (`lo`) | Latin (`la`) | Latvian (`lv`) | Lithuanian (`lt`) |
| Luxembourgish (`lb`) | Macedonian (`mk`) | Maithili (`mai`) | Malagasy (`mg`) |
| Malay (`ms`) | Malayalam (`ml`) | Marathi (`mr`) | Mongolian (`mn`) |
| Nepali (`ne`) | Norwegian, Bokmal (`nb`) | Norwegian, Nynorsk (`nn`) | Odia (`or`) |
| Pashto (`ps`) | Persian (`fa`) | Polish (`pl`) | Portuguese (`pt`) |
| Punjabi (`pa`) | Romanian (`ro`) | Russian (`ru`) | Serbian (`sr`) |
| Sindhi (`sd`) | Sinhala (`si`) | Slovak (`sk`) | Slovenian (`sl`) |
| Spanish (`es`) | Swahili (`sw`) | Swedish (`sv`) | Tamil (`ta`) |
| Telugu (`te`) | Thai (`th`) | Turkish (`tr`) | Ukrainian (`uk`) |
| Urdu (`ur`) | Vietnamese (`vi`) | | |

Gemini detects the spoken language automatically. `en-US` is also advertised
as a Home Assistant compatibility alias for English. See Google's current
[supported language list](https://ai.google.dev/gemini-api/docs/speech-generation#supported-languages).

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
