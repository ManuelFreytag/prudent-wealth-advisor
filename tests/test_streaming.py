import json
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessageChunk
from src.prudent_wealth.api.streaming import stream_response
from src.prudent_wealth.api.schemas import ChatCompletionChunk


@pytest.mark.asyncio
async def test_openai_schema_compliance():
    """Verify that all generated chunks are valid ChatCompletionChunk objects."""
    mock_graph = MagicMock()

    async def mock_astream(*args, **kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "Hello\n"}]),
                {"langgraph_node": "agent"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "thinking", "thinking": "Thinking\n"}]),
                {"langgraph_node": "agent"},
            ),
        )

    mock_graph.astream = mock_astream

    async for line in stream_response(mock_graph, [], "thread-1"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            json_str = line[6:].strip()
            # This will raise a pydantic.ValidationError if not compliant
            chunk_data = json.loads(json_str)
            ChatCompletionChunk(**chunk_data)


@pytest.mark.asyncio
async def test_ignore_nodes_filtering():
    """Verify that messages from ignored nodes are not streamed."""
    mock_graph = MagicMock()

    async def mock_astream(*args, **kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "Skip this"}]),
                {"langgraph_node": "router"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "Keep this\n"}]),
                {"langgraph_node": "agent"},
            ),
        )

    mock_graph.astream = mock_astream

    chunks = []
    async for line in stream_response(mock_graph, [], "thread-1", ignore_nodes=["router"]):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "Keep this\n"
    assert "Skip" not in str(chunks)


@pytest.mark.asyncio
async def test_interleaved_content_and_reasoning():
    """Verify correct handling of alternating text and thinking chunks."""
    mock_graph = MagicMock()

    async def mock_astream(*args, **kwargs):
        # 1. Reasoning
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "thinking", "thinking": "Let me think...\n"}]),
                {"langgraph_node": "agent"},
            ),
        )
        # 2. Content
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "I have an answer.\n"}]),
                {"langgraph_node": "agent"},
            ),
        )

    mock_graph.astream = mock_astream

    chunks = []
    async for line in stream_response(mock_graph, [], "thread-1"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    assert len(chunks) == 3
    assert chunks[0]["choices"][0]["delta"]["reasoning_content"] == "Let me think...\n"
    assert chunks[1]["choices"][0]["delta"]["content"] == "I have an answer.\n"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "role" not in chunks[1]["choices"][0]["delta"]


@pytest.mark.asyncio
async def test_multi_line_buffering():
    """Verify that chunks are yielded only at newlines or end of stream."""
    mock_graph = MagicMock()

    async def mock_astream(*args, **kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "Part 1"}]),
                {"langgraph_node": "agent"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": " - Part 2\nNext line"}]),
                {"langgraph_node": "agent"},
            ),
        )

    mock_graph.astream = mock_astream

    chunks = []
    async for line in stream_response(mock_graph, [], "thread-1"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    assert len(chunks) == 3
    assert chunks[0]["choices"][0]["delta"]["content"] == "Part 1 - Part 2\n"
    assert chunks[1]["choices"][0]["delta"]["content"] == "Next line"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_empty_or_non_ai_chunks():
    """Verify that non-message chunks or empty ones are safely ignored."""
    mock_graph = MagicMock()

    async def mock_astream(*args, **kwargs):
        yield "other_mode", {"something": "else"}
        yield (
            "messages",
            (
                AIMessageChunk(content=[]),
                {"langgraph_node": "agent"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "Valid\n"}]),
                {"langgraph_node": "agent"},
            ),
        )

    mock_graph.astream = mock_astream

    chunks = []
    async for line in stream_response(mock_graph, [], "thread-1"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "Valid\n"


@pytest.mark.asyncio
async def test_stream_response_role_only_in_first_chunk():
    """Verify assistant role is only in the first chunk."""
    mock_graph = MagicMock()

    async def mock_astream(*args, **kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "Hello \n"}]),
                {"langgraph_node": "agent"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": " World\n"}]),
                {"langgraph_node": "agent"},
            ),
        )

    mock_graph.astream = mock_astream

    chunks = []
    async for line in stream_response(mock_graph, [], "thread-1"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "role" not in chunks[1]["choices"][0]["delta"]


@pytest.mark.asyncio
async def test_stream_response_chunk_split_after_newline():
    """Verify correct splitting when LLM chunk contains multiple lines."""
    mock_graph = MagicMock()

    async def mock_astream(*args, **kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "Hello \n*"}]),
                {"langgraph_node": "agent"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": " World\n"}]),
                {"langgraph_node": "agent"},
            ),
        )

    mock_graph.astream = mock_astream

    chunks = []
    async for line in stream_response(mock_graph, [], "thread-1"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello \n"
    assert chunks[1]["choices"][0]["delta"]["content"] == "* World\n"
