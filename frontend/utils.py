"""Utility functions for communicating with the deployed fraud API."""

from __future__ import annotations

from typing import Any

import requests

BASE_URL = "https://fraud-detection-api-wb1m.onrender.com"
PREDICT_ENDPOINT = f"{BASE_URL}/predict"
REQUEST_TIMEOUT_SECONDS = 15


def predict(data: dict[str, Any]) -> dict[str, Any]:
    """Send prediction payload to deployed API and normalize response format."""
    try:
        response = requests.post(PREDICT_ENDPOINT, json=data, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "data": None,
            "error": "timeout",
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "data": None,
            "error": "connection_error",
        }
    except requests.exceptions.HTTPError as exc:
        return {
            "success": False,
            "data": None,
            "error": f"http_error: {exc}",
        }
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "data": None,
            "error": f"request_error: {exc}",
        }

    try:
        response_json = response.json()
    except ValueError:
        return {
            "success": False,
            "data": None,
            "error": "invalid_json_response",
        }

    return {
        "success": True,
        "data": response_json,
        "error": None,
    }
