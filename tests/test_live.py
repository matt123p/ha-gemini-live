"""Tests for the provider-neutral live contract."""

from gemini_live.live import LiveConfig, LiveEvent


def test_live_config_defaults_preserve_legacy_behaviour():
    config = LiveConfig(model="m", voice="v", system_instruction="s")
    assert config.support_barge_in is False
    assert config.transcribe_output is True


def test_live_event_interruption_fields_default_off():
    event = LiveEvent()
    assert event.interrupted is False
    assert event.user_activity_started is False
    assert event.user_activity_stopped is False
    assert event.turn_complete is False


def test_live_event_carries_combined_interruption_and_turn_complete():
    event = LiveEvent(interrupted=True, turn_complete=True)
    assert event.interrupted is True
    assert event.turn_complete is True
