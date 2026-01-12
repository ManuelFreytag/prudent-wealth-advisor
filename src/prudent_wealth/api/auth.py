"""API authentication."""

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import settings

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """Verify the bearer token matches the configured API token.

    Args:
        credentials: HTTP Bearer credentials from the request.

    Returns:
        The verified token.

    Raises:
        HTTPException: If the token is invalid.
    """
    if credentials.credentials != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials
