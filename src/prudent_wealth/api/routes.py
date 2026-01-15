"""FastAPI route handlers."""

import time
import uuid
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from ..agent.graph import create_agent_graph
from .auth import verify_token
from .database import get_checkpointer
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
    ChatMessage,
)
from .streaming import stream_response

router = APIRouter()

# Cached agent graphs by temperature
_agent_graphs: dict[float | None, object] = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_agent(temperature: float | None = None):
    """Get or create an agent graph for the given temperature."""
    if temperature not in _agent_graphs:
        checkpointer = await get_checkpointer()
        _agent_graphs[temperature] = create_agent_graph(
            checkpointer=checkpointer, temperature=temperature
        )
    return _agent_graphs[temperature]


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    token: str = Depends(verify_token),
):
    """OpenAI-compatible chat completion endpoint.

    Supports both streaming and non-streaming responses.
    Uses the `user` field as thread_id for conversation persistence.
    """
    logger.info(request)
    graph = await get_agent(temperature=request.temperature)

    # Convert OpenAI messages to LangChain format
    lc_messages = []
    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        # Note: System and assistant messages could be handled if needed

    # Use user ID as thread_id, or generate one
    thread_id = request.user or f"anonymous-{uuid.uuid4().hex[:8]}"

    if request.stream:
        logger.info("Streaming response")
        return StreamingResponse(
            stream_response(graph, lc_messages, thread_id, request.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        logger.info("Non-streaming response")
        # Non-streaming response
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke({"messages": lc_messages}, config=config)

        # Extract the final assistant message
        final_message = result["messages"][-1] if result["messages"] else None
        content = final_message.content[-1]["text"] if final_message else ""

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(),
        )


@router.get("/v1/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "prudent-wealth-steward",
                "object": "model",
                "created": 1700000000,
                "owned_by": "manuel.freytag26@gmail.com",
            }
        ],
    }
