"""OpenAI-compatible API schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """OpenAI-compatible message format."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "prudent-wealth-steward"
    messages: list[ChatMessage]
    stream: bool = True
    user: str | None = Field(
        default=None, description="User ID, used as thread_id for conversation persistence"
    )
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    max_tokens: int | None = None


class DeltaContent(BaseModel):
    """Delta content in streaming response."""

    content: str | None = None
    reasoning_content: str | None = None
    role: str | None = None


class ChatCompletionChoice(BaseModel):
    """Single choice in completion response."""

    index: int = 0
    delta: DeltaContent | None = None
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """Streaming response chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Non-streaming response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None
