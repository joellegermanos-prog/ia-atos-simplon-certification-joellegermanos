"""HTTP adapter for the agency user registry."""
from __future__ import annotations

import os
from urllib.parse import quote

import httpx


class UserRegistryError(RuntimeError):
    """Raised when the user registry cannot validate an applicant."""


class UserRegistryClient:
    """Small configurable adapter for the transactional user repository."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("USER_REGISTRY_URL", "")).rstrip("/")
        self.timeout = float(os.environ.get("USER_REGISTRY_TIMEOUT_SECONDS", "2"))

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    async def ensure_user_exists(self, user_id: str) -> None:
        """Validate the user against the configured registry before scoring."""
        if not self.enabled:
            raise UserRegistryError("USER_REGISTRY_URL is not configured")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/users/{quote(user_id, safe='')}"
                )
        except httpx.RequestError as exc:
            raise UserRegistryError("User registry unavailable") from exc
        if response.status_code == 404:
            raise UserRegistryError("Unknown user_id")
        if response.status_code >= 400:
            raise UserRegistryError(
                f"User registry returned HTTP {response.status_code}"
            )
