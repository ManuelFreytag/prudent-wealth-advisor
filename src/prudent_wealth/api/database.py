"""Database checkpointer management for conversation persistence."""

from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ..config import settings

# Global checkpointer instance
_checkpointer: BaseCheckpointSaver | None = None

async def get_checkpointer():
    """Returns the globally active checkpointer."""
    global _checkpointer
    if _checkpointer is None:
        # Fallback for local/testing if lifespan didn't run
        return MemorySaver()
    return _checkpointer

# New helper specifically for lifespan
def get_postgres_checkpointer_context(url: str):
    return AsyncPostgresSaver.from_conn_string(url)
    
def set_global_checkpointer(cp):
    global _checkpointer
    _checkpointer = cp
