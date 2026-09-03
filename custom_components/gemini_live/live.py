"""Provider-neutral contract for realtime model sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class LiveTool:
    """A function exposed to a realtime model."""

    name: str
    description: str
    parameters: dict[str, Any] | None = None


@dataclass(slots=True)
class LiveConfig:
    """Settings shared by all realtime providers."""

    model: str
    voice: str
    system_instruction: str
    tools: list[LiveTool] = field(default_factory=list)
    transcribe_output: bool = True


@dataclass(slots=True)
class LiveToolCall:
    """A function call requested by a realtime model."""

    name: str
    call_id: str | None
    arguments: dict[str, Any]


@dataclass(slots=True)
class LiveToolResponse:
    """The local result of a realtime model function call."""

    name: str
    call_id: str | None
    response: Any


@dataclass(slots=True)
class LiveEvent:
    """One normalized event emitted by a realtime provider."""

    audio: bytes | None = None
    text: str | None = None
    input_transcript: str | None = None
    output_transcript: str | None = None
    tool_calls: list[LiveToolCall] = field(default_factory=list)
    turn_complete: bool = False
    go_away: Any = None
    session_resumption_update: Any = None


class LiveSession(Protocol):
    """Operations used by the Home Assistant pipeline."""

    @property
    def is_open(self) -> bool: ...

    async def send_audio(self, audio: bytes) -> None: ...

    async def end_audio(self) -> None: ...

    async def send_text(self, text: str) -> None: ...

    async def send_tool_responses(
        self, responses: list[LiveToolResponse]
    ) -> None: ...

    def receive(self) -> AsyncIterator[LiveEvent]: ...


class LiveClient(Protocol):
    """Factory for configured provider sessions."""

    def connect(self, config: LiveConfig) -> Any: ...
