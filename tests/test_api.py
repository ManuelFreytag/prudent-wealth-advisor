"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_service_info(self, client):
        """Test root endpoint returns service information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Prudent Wealth Steward"
        assert "version" in data


class TestAuthentication:
    """Tests for API authentication."""

    def test_missing_auth_header(self, client):
        """Test request without auth header is rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        # HTTPBearer returns 401 when credentials are missing but auto_error=True
        assert response.status_code in (401, 403)

    def test_invalid_token(self, client):
        """Test request with invalid token is rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401

    @pytest.mark.skip(reason="Requires valid GEMINI_API_KEY to test full flow")
    def test_valid_token_accepted(self, client, auth_headers):
        """Test request with valid token passes auth and returns response."""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
            headers=auth_headers,
        )
        assert response.status_code == 200


class TestModelsEndpoint:
    """Tests for models listing endpoint."""

    def test_list_models(self, client, auth_headers):
        """Test listing available models."""
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "prudent-wealth-steward"
