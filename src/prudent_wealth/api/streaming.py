"""SSE streaming utilities using native LangGraph stream modes.

This module provides functions to parse agent message chunks and stream them
as Server-Sent Events (SSE) compatible with the OpenAI API format.
"""

import time
import uuid
import logging
import re
from collections.abc import AsyncGenerator
from typing import Optional

from langchain_core.messages import AIMessageChunk, BaseMessage
from .schemas import ChatCompletionChunk, ChatCompletionChoice, DeltaContent

logger = logging.getLogger(__name__)
 
 CHAT_COMPLETION_PREFIX = "chatcmpl-"


def parse_message_chunk(chunk: BaseMessage) -> Optional[ChatCompletionChoice]:
    """
    Parses a LangChain message chunk into a DeltaContent object.

    Handles both standard string content and structured content (e.g., Gemini's
    thinking and text blocks). Buffering and newline handling are managed
    by the caller.

    Args:
        chunk: The message chunk emitted by the LLM (typically AIMessageChunk).

    Returns:
        ChatCompletionChoice if valid content (text or reasoning) exists, else None.
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
                    reasoning_content=re.sub(r"\n{1,}", "\\n", item["thinking"]),
                    role="assistant",
                )

            # Case A2: Text block
            elif item.get("type") == "text":
                # avoid double newlines which signals new chunk
                delta = DeltaContent(
                    content=re.sub(r"\n{1,}", "\\n", item["text"]),
                    role="assistant",
                )

            else:
                # Unknown block type, skip
                return None

            return ChatCompletionChoice(delta=delta, finish_reason=None)


async def stream_response(
    graph,
    messages: list,
    thread_id: str,
    model_name: str = "prudent-wealth-steward",
    ignore_nodes: list | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream agent responses using LangGraph's native 'messages' mode.

    Coordinates the execution of a LangGraph agent and streams its output
    as OpenAI-compatible SSE chunks. It includes internal buffering logic
     to ensure that chunks are yielded at logical boundaries (e.g., newlines)
     to prevent UI stuttering or broken formatting.

    Args:
        graph: The compiled LangGraph agent.
        messages: List of LangChain messages providing the conversation history.
        thread_id: Unique identifier for the conversation thread.
        model_name: The name of the model to display in the response chunks.
        ignore_nodes: List of node names whose messages should be excluded from the stream.

    Yields:
        OpenAI-compatible SSE data strings ("data: {...}\n\n").
    """
    response_id = f"{CHAT_COMPLETION_PREFIX}{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    if ignore_nodes is None:
        ignore_nodes = []

    is_first_chunk = True
    content_buffer = ""
    reasoning_buffer = ""

    async def yield_from_buffer(buffer_text: str, field_name: str, is_final: bool = False):
        nonlocal is_first_chunk

        # If final, yield everything
        if is_final:
            to_yield = buffer_text
            remaining = ""
        else:
            # Find last escaped newline (\n)
            last_newline_idx = buffer_text.rfind("\n")
            if last_newline_idx == -1:
                # No escaped newline, just return the buffer as remaining
                yield None, buffer_text
                return
            # Include the \n in the yielded content (+1 for the character)
            to_yield = buffer_text[: last_newline_idx + 1]
            remaining = buffer_text[last_newline_idx + 1 :]

        if to_yield:
            delta_kwargs = {field_name: to_yield + " "}
            if is_first_chunk:
                delta_kwargs["role"] = "assistant"
                is_first_chunk = False

            delta = DeltaContent(**delta_kwargs)
            choice = ChatCompletionChoice(delta=delta, index=0, finish_reason=None)
            completion_chunk = ChatCompletionChunk(
                id=response_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model_name,
                choices=[choice],
            )
            yield "data: " + completion_chunk.model_dump_json(
                exclude_unset=True
            ) + "\n\n", remaining
        else:
            yield None, remaining

    async for mode, payload in graph.astream(
        {"messages": messages},
        config=config,
        stream_mode=["messages"],
    ):
        # Payload in 'messages' mode is a tuple: (chunk, metadata)
        chunk, node_info = payload

        # Ignore streaming messages from the router node
        if node_info.get("langgraph_node") not in ignore_nodes:
            choice = parse_message_chunk(chunk)

            # If a valid delta was parsed, buffer it
            if choice:
                if choice.delta.content:
                    content_buffer += choice.delta.content
                    async for sse, remaining in yield_from_buffer(content_buffer, "content"):
                        if sse:
                            logger.info(sse)
                            yield sse
                        content_buffer = remaining

                if choice.delta.reasoning_content:
                    reasoning_buffer += choice.delta.reasoning_content
                    async for sse, remaining in yield_from_buffer(
                        reasoning_buffer, "reasoning_content"
                    ):
                        if sse:
                            logger.info(sse)
                            yield sse
                        reasoning_buffer = remaining

    # Flush remaining buffers
    if content_buffer:
        async for sse, _ in yield_from_buffer(content_buffer, "content", is_final=True):
            logger.info(sse)
            if sse:
                yield sse
    if reasoning_buffer:
        async for sse, _ in yield_from_buffer(reasoning_buffer, "reasoning_content", is_final=True):
            logger.info(sse)
            if sse:
                yield sse

    # Send final [DONE] event
    choice = ChatCompletionChoice(delta=DeltaContent(), index=0, finish_reason="stop")
    completion_chunk = ChatCompletionChunk(
        id=response_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model_name,
        choices=[choice],
    )

    yield "data: " + completion_chunk.model_dump_json(exclude_unset=True) + "\n\n"
    yield "data: [DONE]\n\n"
