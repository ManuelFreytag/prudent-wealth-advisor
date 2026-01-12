"""SSE streaming utilities for OpenAI-compatible responses."""

import time
import uuid
from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessageChunk
from .schemas import ChatCompletionChunk, ChatCompletionChoice, DeltaContent


def get_delta_content(kind: str, event: dict) -> DeltaContent:
    """Get delta content from event."""
    if kind == "on_chat_model_stream":
        # LLM token output
        chunk = event["data"].get("chunk")
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            return DeltaContent(content=chunk.content[0]["text"], role="assistant")
    elif kind == "on_chain_stream":
        # status updates (for transparency)
        return DeltaContent(content=f"<think>{event['name']}</think>", role="assistant")
    elif kind == "on_tool_start":
        # stream tool calls (for transparency)
        tool_name = event.get("name", "tool")
        return DeltaContent(content=f"<think>{tool_name}</think>", role="assistant")
    return DeltaContent()


async def stream_response(
    graph,
    messages: list,
    thread_id: str,
    model_name: str = "prudent-wealth-steward",
) -> AsyncGenerator[str, None]:
    """Stream agent responses as Server-Sent Events.

    Args:
        graph: The compiled LangGraph agent.
        messages: List of LangChain messages to process.
        thread_id: Thread ID for conversation persistence.
        model_name: Model name to include in response chunks.

    Yields:
        SSE-formatted chunks compatible with OpenAI streaming format.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    config = {"configurable": {"thread_id": thread_id}}

    # Stream events from the graph
    async for event in graph.astream_events(
        {"messages": messages},
        config=config,
        version="v2",
    ):
        # Stream events from the graph
        kind = event["event"]
        choice = ChatCompletionChoice(delta=get_delta_content(kind, event))
        completion_chunk = ChatCompletionChunk(
            id=response_id, created=created, model=model_name, choices=[choice]
        )
        yield "data: " + completion_chunk.model_dump_json() + "\n\n"

    # Send finish chunk
    choice = ChatCompletionChoice(delta=DeltaContent(), finish_reason="stop")
    completion_chunk = ChatCompletionChunk(
        id=response_id, created=created, model=model_name, choices=[choice]
    )

    yield "data: " + completion_chunk.model_dump_json() + "\n\n"
    yield "data: [DONE]\n\n"
