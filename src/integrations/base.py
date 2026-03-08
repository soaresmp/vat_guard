"""
Base Integration Client
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class BaseIntegrationClient:
    """
    Async HTTP client base class for external system integrations.
    Handles authentication, retries, and error normalisation.
    """

    DEFAULT_TIMEOUT = 30

    def __init__(self, base_url: str, api_key: str, timeout: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        try:
            client = self._get_client()
            response = await client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Integration GET %s returned %s: %s",
                path, exc.response.status_code, exc.response.text[:200]
            )
            return None
        except Exception as exc:
            logger.error("Integration GET %s failed: %s", path, exc)
            return None

    async def _post(self, path: str, data: dict) -> dict | None:
        try:
            client = self._get_client()
            response = await client.post(path, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Integration POST %s returned %s: %s",
                path, exc.response.status_code, exc.response.text[:200]
            )
            return None
        except Exception as exc:
            logger.error("Integration POST %s failed: %s", path, exc)
            return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
