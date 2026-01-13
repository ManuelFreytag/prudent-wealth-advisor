"""SSE streaming utilities using native LangGraph stream modes."""

import time
import uuid
from collections.abc import AsyncGenerator
from typing import Optional

from langchain_core.messages import AIMessageChunk, BaseMessage
from .schemas import ChatCompletionChunk, ChatCompletionChoice, DeltaContent


def parse_message_chunk(chunk: BaseMessage) -> Optional[ChatCompletionChoice]:
    """
    Parses a LangChain message chunk into a DeltaContent object.
    Handles both standard string content and structured content (e.g., Gemini).

    Args:
        chunk: The message chunk emitted by the LLM.

    Returns:
        ChatCompletionChoice if valid content exists, else None.
    """
    # Only process AIMessageChunks with actual content
    if not isinstance(chunk, AIMessageChunk) or not chunk.content:
        return None

    # Case A: Complex content (list of dicts) - used by Gemini
    if isinstance(chunk.content, list):
        item = chunk.content[0]
        if isinstance(item, dict):

            # Case A1: Thinking block
            if item.get("type") == "thinking":
                # Format thinking blocks inside tags for the UI
                delta = DeltaContent(
                    reasoning_content=item["thinking"].replace("\n\n", "\n"),
                    role="assistant",
                )

            # Case A2: Text block
            elif item.get("type") == "text":
                # avoid double newlines which signals new chunk
                delta = DeltaContent(
                    content=item["text"].replace("\n\n", "\n"),
                    role="assistant",
                )

            return ChatCompletionChoice(delta=delta)


async def stream_response(
    graph,
    messages: list,
    thread_id: str,
    model_name: str = "prudent-wealth-steward",
) -> AsyncGenerator[str, None]:
    """Stream agent responses using LangGraph's native 'updates' and 'messages' modes.

    Args:
        graph: The compiled LangGraph agent.
        messages: List of LangChain messages to process.
        thread_id: Thread ID for conversation persistence.
        model_name: Model name to include in response chunks.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Stream both tokens (messages) and state changes (updates)
    async for mode, payload in graph.astream(
        {"messages": messages},
        config=config,
        stream_mode=["messages"],
    ):
        created = int(time.time())
        # Payload in 'messages' mode is a tuple: (chunk, metadata)
        chunk, node_info = payload

        # Ignore streaming messages from the router node
        if node_info.get("langgraph_node") != "router":
            choice = parse_message_chunk(chunk)

            # If a valid delta was parsed, yield it as SSE
            if choice:
                completion_chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=model_name,
                    choices=[choice],
                )
                yield "data: " + completion_chunk.model_dump_json() + "\n\n"

    # Send final [DONE] event
    choice = ChatCompletionChoice(delta=DeltaContent(), finish_reason="stop")
    completion_chunk = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=model_name,
        choices=[choice],
    )

    yield "data: " + completion_chunk.model_dump_json() + "\n\n"
    yield "data: [DONE]\n\n"
