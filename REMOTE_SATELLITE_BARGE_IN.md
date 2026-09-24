# Remote Satellite Barge-In

This document describes the end-to-end barge-in path used by
`ha-gemini-live`, the required Home Assistant Core changes, and the behavior
required from an ESPHome voice satellite.

Barge-in lets a user speak while the assistant is playing a response. The old
response must stop promptly, microphone audio must continue reaching the live
model, and replacement audio must play without starting a new conversation
session.

## Requirements

Barge-in requires all of the following:

- the **Support barge-in** option enabled in `ha-gemini-live`;
- Home Assistant Core with the TTS interruption API and ESPHome playback path;
- a full-duplex satellite that continues microphone capture during playback;
- effective acoustic echo cancellation (AEC), or headphones; and
- ESPHome firmware that advertises and implements the playback-flush extension.

The integration checks for the required Core API before allowing barge-in to be
enabled. Specifically, Core must provide:

- `TTSAudioRequest.on_audio_interrupt`; and
- `TextToSpeechEntity.supports_audio_interrupt`.

If either API is missing, the barge-in option is hidden from the config and
options flows, and any stored barge-in setting is ignored at runtime. With
barge-in disabled, the integration continues to work with an unmodified Home
Assistant Core.

## End-to-end interruption path

```text
User speaks during assistant playback
    -> satellite continues sending microphone PCM
    -> Gemini/OpenAI detects speech and interrupts generation
    -> ha-gemini-live discards queued model audio
    -> Gemini Live TTS invokes Core's on_audio_interrupt callback
    -> Core sends TTS_STREAM_START with {"flush": "1"}
    -> ESPHome synchronously purges old playback without ending the response
    -> replacement model audio continues in the same TTS response
```

The provider is authoritative about whether an interruption occurred. Local
speech detection on the satellite can stop its speaker sooner, but it does not
replace the provider-to-Core interruption path.

## Responsibilities

### Gemini Live integration and provider

The integration and live-model provider:

- keep forwarding microphone PCM while the model is speaking;
- use provider-side voice activity detection to recognize an interruption;
- cancel or abandon the interrupted model response;
- discard assistant audio still queued inside the integration;
- notify Home Assistant through the TTS interruption callback;
- stream replacement audio through the existing TTS response; and
- keep the live-model session available so conversational context is retained.

The response-transcription option is independent of barge-in. Enabling
barge-in must not implicitly enable or display the assistant transcript.

### Home Assistant Core

The Core patch:

- lets an entity opt in through `supports_audio_interrupt`;
- preserves the raw input until either a live or cached consumer claims it;
- bypasses memory and disk caches for one live interrupt-capable consumer;
- keeps URL, media-player, VoIP, and unsupported-satellite consumers on the
  normal cached path;
- passes the satellite's `on_audio_interrupt` callback directly to the engine;
- passes the consumer's requested output options to the TTS engine; and
- prevents a competing consumer from starting a second model session.

An interruptible engine must return PCM WAV, preserve the original WAV header
across interruptions, discard its pending old-response audio before invoking
the callback, and yield only replacement audio afterward. Because the live path
bypasses Core's normal audio conversion, the engine must advertise and honor
the requested format, sample rate, channel count, and sample width. ESPHome
currently requests 16 kHz, mono, 16-bit PCM WAV, which `ha-gemini-live` emits.

ESPHome playback starts at `INTENT_PROGRESS` for interruptible streams, using
the token supplied at `RUN_START`, only when the device advertises both speaker
support and the flush extension. Other consumers retain the previous behavior.

### Remote satellite

The satellite must:

- capture and transmit microphone audio while speaker audio is playing;
- run AEC so assistant audio is not mistaken for user speech;
- advertise feature bit `1 << 7` in its voice-assistant feature flags;
- recognize `VOICE_ASSISTANT_TTS_STREAM_START` with `flush=1` as a flush of the
  current response, not the start of a separate response;
- synchronously purge all old playback and AEC reference data;
- keep the overall Assist pipeline alive across the interruption; and
- report playback completion only after the final stream end.

## ESPHome protocol extension

The extension uses an existing event with new data:

```text
VOICE_ASSISTANT_TTS_STREAM_START {"flush": "1"}
<replacement TTS audio>
```

Core sends this only to a device whose voice-assistant feature flags contain
bit `1 << 7`. Firmware without the bit stays on the pre-existing cached TTS
path and never receives a mid-response flush. This capability gate preserves
legacy ESPHome behavior.

The ESPHome voice assistant must inspect the start-event data and synchronously
clear the speaker playback queue when `flush` is `1`. The clear must happen in
the voice-assistant event handler before later audio packets can be consumed; a
deferred YAML automation is not sufficient to guarantee ordering.

For the `aec_speexdsp` speaker, the handler should synchronously call
`clear_playback()`, which clears the playback and AEC reference buffers and
resets resampler state. The flush event must not finish the announcement or
Assist response. Core sends the ordinary `VOICE_ASSISTANT_TTS_STREAM_END` once,
after the final audio.

Core deliberately does not send an intermediate `TTS_STREAM_END`. Existing
ESPHome firmware treats it as end-of-stream and drains buffered audio rather
than discarding it.

## Unsupported transports

Wyoming interruptible playback is not part of the current Core patch.
Wyoming's `AudioStop` means end-of-audio and may produce `Played`; using it as a
flush can finish the Home Assistant response while replacement audio is still
playing. A future Wyoming implementation needs an explicit,
capability-negotiated flush operation before it can safely use this path.

## Microphone and echo cancellation

Capture and playback must run in separate tasks or threads. Receiving assistant
audio must never pause or close microphone capture.

Send 16 kHz, 16-bit, mono PCM from the microphone for the complete pipeline
run. A microphone packet duration of roughly 20-40 ms is recommended; larger
packets add directly to interruption latency.

Without effective AEC, provider VAD can hear the assistant's speaker output and
cause repeated self-interruptions. Feed the exact rendered speaker samples into
the AEC reference path and account for hardware, operating-system, decoder, and
speaker buffering. Noise suppression and automatic gain control can help after
AEC, but do not replace it.

For hardware without reliable AEC, use headphones, a push-to-interrupt control,
or leave barge-in disabled.

## Playback behavior

When local speech is detected during playback, a capable satellite may stop and
purge playback immediately rather than waiting for the provider round trip. It
must continue transmitting microphone audio.

When the Core flush-marked stream-start event arrives, the satellite must:

1. stop the audio device;
2. clear device/DMA, decoder, jitter, application, and AEC reference queues;
3. reset playback and resampler state for the replacement audio;
4. continue microphone transmission; and
5. accept replacement audio as part of the same TTS response.

The underlying Home Assistant TTS result remains open. The flush event neither
closes nor restarts the Assist pipeline.

## Suggested satellite state machine

```text
LISTENING
    user utterance ends
        -> PROCESSING

PROCESSING
    first assistant audio/start event arrives
        -> SPEAKING

SPEAKING
    local user speech starts
        -> optionally purge playback immediately
        -> keep transmitting microphone audio
        -> LISTENING

SPEAKING or LISTENING
    Core flush-marked stream-start arrives
        -> purge old playback
        -> prepare a clean replacement segment
        -> PROCESSING

PROCESSING
    replacement audio arrives
        -> SPEAKING

SPEAKING
    final stream-end and playback completion
        -> IDLE or LISTENING, according to pipeline continuation
```

The local state machine may react before Core, but the Core signal is the
authoritative boundary between old and replacement assistant audio.

## Buffering and latency

Suggested starting targets:

| Component | Target |
|---|---:|
| Microphone packet duration | 20-40 ms |
| Local playback queue | 40-100 ms |
| Speech-start debounce | 40-100 ms |
| Local playback purge after speech start | Under 50 ms |
| Integration response-audio buffer | About 200 ms |

The integration bounds its response-audio queue. On interruption it clears
that queue, Core clears parser, pacing, and pending output state, and the
satellite flush clears audio already delivered to the remote device.

Smaller buffers reduce audible overrun after interruption but increase
sensitivity to network jitter and scheduling delays. Measure latency on the
actual satellite hardware.

## Compatibility behavior

| Barge-in | Core API | ESPHome flush bit | Result |
|---|---|---|---|
| Disabled | Missing or present | Missing or present | Normal legacy TTS behavior |
| Enabled | Present | Present | Live interruptible playback and remote flush |
| Enabled | Present | Missing | Normal cached satellite playback; no remote barge-in |
| Enabled | Missing | Any | Configuration is rejected and setup is blocked |

The custom Core image and patch are release-specific. Do not copy patched Core
files into a different Home Assistant version without rebasing and testing the
patch against that release.

## Failure handling

- On network loss, stop playback, close capture, and return to an error or idle
  state.
- If replacement audio does not arrive within the normal response timeout, end
  the interaction instead of remaining in `PROCESSING` indefinitely.
- If AEC diverges or speaker loopback repeatedly interrupts the model, disable
  automatic interruption and provide a push-to-interrupt fallback.
- If the satellite stops microphone transmission during playback, barge-in is
  unavailable for that turn; ordinary playback should still work.

## Validation checklist

Test with response transcription both enabled and disabled.

- Barge-in cannot be enabled on an unpatched Core.
- With barge-in disabled, behavior is unchanged on patched and unpatched Core.
- Microphone packets continue while assistant audio is playing.
- Speaker loopback alone does not trigger interruption.
- Real user speech interrupts the provider response.
- The integration discards its queued old-response audio.
- A capable ESPHome device receives `TTS_STREAM_START` with `flush=1`.
- An incapable ESPHome device uses the normal cached path and receives no
  mid-response flush.
- The satellite purges old buffered playback at the interruption boundary.
- No intermediate `TTS_STREAM_END` or ESPHome completion is reported.
- Replacement audio starts with reset timing and is not clipped.
- Audio with a mismatched rate, channel count, or sample width is rejected
  rather than silently converted in the live path.
- The final stream end produces the normal playback-completed indication.
- Enabling barge-in does not enable response transcription.
- Repeated interruptions do not leak tasks, callbacks, buffers, or response
  state.

For an end-to-end test, request a long answer and interrupt it after several
words. Verify that old speech stops, microphone upload continues, the transport
emits the flush-marked start event, replacement speech plays, and the same
live-model session retains its conversation context.
