# Changelog

All notable changes to Gemini Live for Home Assistant are documented here.

## Unreleased

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
