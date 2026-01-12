"""Pytest fixtures for testing."""

import os

import pytest

# Set test environment variables before importing app modules
os.environ.setdefault("GEMINI_API_KEY", "test-api-key")
os.environ.setdefault("API_TOKEN", "dev-token-12345")


@pytest.fixture
def api_token():
    """Return the test API token."""
    return "dev-token-12345"


@pytest.fixture
def auth_headers(api_token):
    """Return authorization headers for API requests."""
    return {"Authorization": f"Bearer {api_token}"}
