"""JWT token management for client authentication."""

import json
import os
import time
from pathlib import Path
from typing import Optional

from penguincode.shared.interfaces import IAuthService


class TokenManager:
    """Manages JWT token storage and refresh.

    Stores tokens in a local file for persistence across sessions.
    Handles token expiry and automatic refresh.
    """

    def __init__(self, token_path: str = "~/.penguincode/token"):
        self.token_path = Path(token_path).expanduser()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at: float = 0

        # Load existing token if available
        self._load_token()

    def _load_token(self) -> None:
        """Load token from disk."""
        if not self.token_path.exists():
            return

        try:
            with open(self.token_path, "r") as f:
                data = json.load(f)
                self._access_token = data.get("access_token")
                self._refresh_token = data.get("refresh_token")
                self._expires_at = data.get("expires_at", 0)
        except Exception:
            pass

    def _save_token(self) -> None:
        """Save token to disk."""
        try:
            # Create directory if needed
            self.token_path.parent.mkdir(parents=True, exist_ok=True)

            # Save with restricted permissions
            with open(self.token_path, "w") as f:
                json.dump({
                    "access_token": self._access_token,
                    "refresh_token": self._refresh_token,
                    "expires_at": self._expires_at,
                }, f)

            # Set file permissions to owner-only
            os.chmod(self.token_path, 0o600)

        except Exception:
            pass

    def store_token(
        self,
        access_token: str,
        refresh_token: str,
        expires_in: int,
    ) -> None:
        """Store a new token.

        Args:
            access_token: JWT access token
            refresh_token: Refresh token for renewal
            expires_in: Seconds until expiry
        """
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = time.time() + expires_in
        self._save_token()

    def get_token(self) -> Optional[str]:
        """Get current access token if not expired.

        Returns None if token is expired or not available.
        """
        if not self._access_token:
            return None

        # Check if token is expired (with 60s buffer)
        if time.time() > self._expires_at - 60:
            return None

        return self._access_token

    def get_refresh_token(self) -> Optional[str]:
        """Get refresh token for renewal."""
        return self._refresh_token

    def is_expired(self) -> bool:
        """Check if the access token is expired."""
        return time.time() > self._expires_at - 60

    def clear(self) -> None:
        """Clear stored tokens."""
        self._access_token = None
        self._refresh_token = None
        self._expires_at = 0

        if self.token_path.exists():
            self.token_path.unlink()


class LocalAuthService(IAuthService):
    """Authentication service for local mode (no-op).

    Used when running in local mode where no auth is needed.
    """

    async def authenticate(
        self,
        api_key: str,
        client_id: str,
    ) -> Optional[str]:
        """No authentication needed in local mode."""
        return "local-token"

    async def refresh_token(self, refresh_token: str) -> Optional[str]:
        """No token refresh needed in local mode."""
        return "local-token"

    async def validate_token(self, access_token: str) -> bool:
        """All tokens valid in local mode."""
        return True

    def get_token(self) -> Optional[str]:
        """Return dummy token for local mode."""
        return "local-token"
