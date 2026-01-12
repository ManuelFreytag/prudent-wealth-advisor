"""Prudent Wealth Steward - FastAPI Application Entry Point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.prudent_wealth.api import router
from src.prudent_wealth.config import settings
from src.prudent_wealth.api.database import (
    get_postgres_checkpointer_context,
    set_global_checkpointer,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Setup checkpointer with database connection.
    Will spin up the checkpointer on app startup and close it on app shutdown.
    Use context manager to close gracefully.
    """
    if settings.database_url:
        async with get_postgres_checkpointer_context(settings.database_url) as checkpointer:
            await checkpointer.setup()
            set_global_checkpointer(checkpointer)

            yield
            # Connection pool closes automatically when app stops
    else:
        # Development/Memory mode
        from langgraph.checkpoint.memory import MemorySaver

        set_global_checkpointer(MemorySaver())
        yield


app = FastAPI(
    title="Prudent Wealth Steward",
    description="Conservative financial advisor agent API with OpenAI-compatible interface",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "prudent-wealth-steward"}


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Prudent Wealth Steward",
        "version": "0.1.0",
        "description": "Conservative financial advisor agent",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
