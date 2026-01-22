import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import AIMessageChunk
from src.prudent_wealth.api.streaming import stream_response
import json


@pytest.mark.asyncio
async def test_stream_response_role_only_in_first_chunk():
    # Mock graph
    mock_graph = MagicMock()

    # helper to create async generator
    async def mock_astream(*args, **kwargs):
        # yields tuples of (chunk, metadata)
        # Chunk 1
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "Hello \n"}]),
                {"langgraph_node": "agent"},
            ),
        )
        # Chunk 2
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": " World"}]),
                {"langgraph_node": "agent"},
            ),
        )

    mock_graph.astream = mock_astream

    chunks = []
    async for line in stream_response(mock_graph, [], "thread-1"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            json_str = line[6:]
            chunks.append(json.loads(json_str))

    assert len(chunks) == 3

    # Check first chunk
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello \n"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

    # Check second chunk
    assert chunks[1]["choices"][0]["delta"]["content"] == " World"
    assert chunks[1]["choices"][0]["delta"].get("role") is None


@pytest.mark.asyncio
async def test_stream_response_chunk_split_after_newline():
    # Mock graph
    mock_graph = MagicMock()

    # helper to create async generator
    async def mock_astream(*args, **kwargs):
        # yields tuples of (chunk, metadata)
        # Chunk 1
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "Hello \n*"}]),
                {"langgraph_node": "agent"},
            ),
        )
        # Chunk 2
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": " World"}]),
                {"langgraph_node": "agent"},
            ),
        )

    mock_graph.astream = mock_astream

    chunks = []
    async for line in stream_response(mock_graph, [], "thread-1"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            json_str = line[6:]
            chunks.append(json.loads(json_str))

    assert len(chunks) == 3

    # Check first chunk
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello \n"

    # Check second chunk
    assert chunks[1]["choices"][0]["delta"]["content"] == "* World"
