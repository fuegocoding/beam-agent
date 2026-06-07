"""Client for accessing paid brains via the Beam API proxy.

When a user installs a paid brain, the brain data is NOT downloaded locally.
Instead, all queries go through the beam_mind API proxy using an install token.
"""
import json
import os
from typing import Any

import httpx


# ── Configuration ─────────────────────────────────────────────────────

API_URL = os.environ.get("BEAM_API_URL", "https://api.openbeam.me")


class BrainProxyClient:
    """Client for querying a paid brain via the Beam API."""

    def __init__(self, slug: str, token: str, api_url: str | None = None):
        self.slug = slug
        self.token = token
        self.api_url = api_url or API_URL

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def search(
        self,
        query: str,
        trust_level: str = "visitor",
        brain_power: str = "standard",
    ) -> list[dict]:
        """Search the brain for relevant nodes."""
        url = f"{self.api_url}/api/v1/brain-proxy/{self.slug}/search"
        payload = {
            "query": query,
            "trust_level": trust_level,
            "brain_power": brain_power,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                # New marketplace proxy returns full dict with nodes/context
                if isinstance(data, dict) and "nodes" in data:
                    return data
                # Legacy paid-brain proxy returns list under "results"
                return data.get("results", [])
        except httpx.ConnectError:
            raise ConnectionError(
                "Cannot connect to Beam API. Paid brains require an internet connection."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise PermissionError("Invalid or expired install token.")
            elif e.response.status_code == 402:
                raise PermissionError("Purchase is no longer active or subscription expired.")
            elif e.response.status_code == 404:
                raise FileNotFoundError(f"Brain '{self.slug}' not found on marketplace.")
            else:
                raise RuntimeError(f"API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Failed to query brain: {e}")

    def get_soul(self) -> str:
        """Get the SOUL.md content for this brain."""
        url = f"{self.api_url}/api/v1/brain-proxy/{self.slug}/soul"

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.text
        except httpx.ConnectError:
            raise ConnectionError(
                "Cannot connect to Beam API. Paid brains require an internet connection."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise PermissionError("Invalid or expired install token.")
            else:
                raise RuntimeError(f"API error: {e.response.status_code}")

    def get_context(self) -> dict:
        """Get behavioral context for this brain."""
        url = f"{self.api_url}/api/v1/brain-proxy/{self.slug}/context"

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError:
            raise ConnectionError(
                "Cannot connect to Beam API. Paid brains require an internet connection."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise PermissionError("Invalid or expired install token.")
            else:
                raise RuntimeError(f"API error: {e.response.status_code}")

    def ping(self) -> bool:
        """Check if the brain proxy is accessible."""
        try:
            self.get_context()
            return True
        except Exception:
            return False
