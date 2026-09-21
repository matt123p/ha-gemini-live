# Changelog

All notable changes to Gemini Live for Home Assistant are documented here.

## Unreleased

- Fixed Gemini 3.8 sessions with affective dialog failing every voice turn with
  "Request contains an invalid argument" (websocket close 1007) by sending
  `enable_affective_dialog` inside `generation_config`, where the v1beta Live
  API wire proto actually defines it, instead of at the top level of the setup
  message.
- Barge-in no longer fails setup on Home Assistant Core builds without TTS
  interruption support. The option is hidden from the config and options flows,
  any stored barge-in setting is ignored at runtime, and the previous
  `ConfigEntryError` was removed.
- Fixed Gemini Live rejecting the affective-dialog setup with
  "Request contains an invalid argument" (websocket close 1007) by pinning the
  client to the `v1beta` API version when affective dialog is enabled, which
  the Live API requires for `enable_affective_dialog`.
- Extended the affective-dialog switch to `gemini-3.8-live-extended-thinking`;
  per the Live API documentation only Gemini 3.1 Flash Live lacks support.
- Fixed affective-dialog sessions failing SDK validation by sending
  `enable_affective_dialog` at the top level of the Gemini Live config.
- Added `gemini-3.8-live-extended-thinking`, including its asynchronous tool
  declarations, configurable low/medium/high thinking level, and
  multi-utterance interaction lifecycle, and changed model selection to a
  dropdown. Model-specific settings are hidden when they do not apply.
- Added optional native Google Search grounding for every available Gemini Live
  model. This can run alongside Home Assistant Assist function tools and
  replaces the separate search-agent workaround for Gemini entries. When
  enabled, the system instruction now explicitly directs Gemini to use Search
  for current, changing, or explicitly requested online information.
- Preserve the originating Assist pipeline context for voice-triggered tool
  calls, so Home Assistant can retain satellite and pipeline provenance in
  downstream service calls and Logbook activity. Unavailable or invalid
  pipeline contexts continue to use an anonymous fallback.
- Fixed typed text turns failing with "live-model text path returned no usable
  text" when the model ended the conversation via `end_conversation` without
  producing any text. The text path now returns the unique `-- gemini live --`
  pipeline placeholder for that turn instead of the generic error response.

## 1.0.8

- Barge-in now keeps listening for the whole pipeline run: microphone audio
  continues to be forwarded after the model finishes a turn, so the user can
  interrupt while a response is still playing, and a follow-up run reuses the
  same provider session.
- Bound the response audio buffered between the live model and Home Assistant
  to roughly 200 milliseconds with producer backpressure, so a barge-in
  interruption discards almost all queued output and playback stops quickly.
- Preserve the 24-to-16 kHz resampler phase across provider audio packets and
  remove per-packet warning logs, preventing discontinuities and bursty
  playback during long streamed responses.
- Added `gemini-3.8-live` as a model choice. Tool declarations sent to it pin
  synchronous (blocking) function calling, preserving the integration's
  request/execute/respond flow on models that default to asynchronous
  execution.
- Added an Affective dialog switch for Gemini 3.8 models. When enabled, the
  model reads the tone and emotion in the user's voice and adapts its own
  speaking style to match. The switch is hidden for models that do not
  support it.
- Embedded the unique per-turn id in the pipeline placeholder whenever no
  input transcript is available, so Home Assistant's TTS message cache can no
  longer replay an earlier turn's audio.

## 1.0.7

- Encourage Gemini Live and OpenAI GPT Realtime to call `show_text` before
  speaking for detailed or information-dense answers, followed by a concise
  spoken summary.
- Removed the instruction that made short commands such as "stop" prioritize
  stopping an actively ringing alarm or timer.
- Fixed OpenAI GPT Realtime rejecting Home Assistant tools such as
  `HassStartTimer` whose parameter schemas use a top-level union.

## 1.0.6

- Added OpenAI GPT Realtime as a provider option with provider-specific models,
  voices, credentials, response transcription, and native audio streaming.
- Factored the live-model protocol into a provider-neutral contract, with peer
  Gemini SDK and OpenAI WebSocket adapters sharing the conversation, Assist tool,
  session, STT, and TTS pipeline logic.
- Kept existing config entries backward compatible by treating entries without
  a provider field as Google Gemini.
- Close provider sessions immediately after `end_conversation`, and evict sessions
  when their audio sender or response receiver reports a transport failure.
- Use the existing `-- gemini live --` pipeline placeholder for both providers so
  remote satellites can filter one consistent marker.
- Make OpenAI GPT Realtime call `end_conversation` before a farewell, so Home
  Assistant reliably stops listening when the user finishes the conversation.
- Tune OpenAI GPT Realtime responses for a balanced voice style: direct and
  concise by default, while preserving useful context and detail when needed.

## 1.0.5

- Fixed Home Assistant script parameters being omitted from Gemini tool
  declarations on newer Home Assistant releases.
- Retained schema-conversion compatibility with older Home Assistant releases.

## 1.0.4

- Added the satellite's device ID to the LLM context for voice turns so Home
  Assistant tells Gemini which area the microphone is in, and lets generic
  commands like "turn on the lights" target that area automatically.
- Added a user-visible response when the configured Google AI project exceeds
  its monthly spending cap, instead of failing the Assist pipeline silently.
- Added a persistent Home Assistant Repairs issue with a link to Google AI
  Studio. The issue clears automatically after Gemini Live reconnects.
- Added detailed debug logging for LLM tool calls and their responses in both
  voice and typed conversation paths.
- Removed obsolete instructions for patching Home Assistant Core now that the
  low-latency change is included upstream.

## 1.0.3

- Added an optional `show_text` tool so Gemini can display formatted text when
  response transcription is disabled.
- Kept conversation completion state isolated to the specific conversation
  that invoked `end_conversation`.
- Expanded setup and troubleshooting documentation.

## 1.0.2

- Added an `end_conversation` callback that lets Gemini tell Home Assistant when
  to stop listening for follow-up requests. Completion state is tracked
  independently for each conversation.
- Made short opening commands such as "stop" prioritize stopping an actively
  ringing alarm or timer before ending the conversation.
- Documented the Home Assistant Core custom-component override that reduces
  response latency on ESPHome Assist satellites.

## 1.0.1

- Fixed HACS and Hassfest validation metadata.

## 1.0.0

- Added Gemini Live speech-to-text, conversation, and cached native-audio
  text-to-speech entities.
- Added HACS metadata, brand assets, translations, validation workflow, and
  installation documentation.
