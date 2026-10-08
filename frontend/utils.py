"""Utility functions for communicating with the deployed fraud API."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_BASE_URL = "https://fraud-detection-api-wb1m.onrender.com"
BASE_URL = os.environ.get("FRAUD_API_URL", DEFAULT_BASE_URL).rstrip("/")
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
        status_code = getattr(exc.response, "status_code", None)
        error_msg = f"http_error ({status_code}): {exc}" if status_code else f"http_error: {exc}"
        return {
            "success": False,
            "data": None,
            "error": error_msg,
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

