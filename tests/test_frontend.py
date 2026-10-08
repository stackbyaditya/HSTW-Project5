from __future__ import annotations

from unittest.mock import MagicMock, patch
import requests

from frontend.utils import predict, BASE_URL


def test_predict_success() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "fraud_probability": 0.12,
        "prediction": 0,
        "label": "LEGITIMATE",
    }
    mock_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_response) as mock_post:
        payload = {
            "ip": 87540,
            "app": 12,
            "device": 1,
            "os": 13,
            "channel": 497,
            "click_time": "2017-11-07 09:30:38",
        }
        result = predict(payload)

        assert result["success"] is True
        assert result["data"]["label"] == "LEGITIMATE"
        assert result["error"] is None
        mock_post.assert_called_once_with(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=15,
        )


def test_predict_timeout() -> None:
    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        result = predict({"ip": 123})
        assert result["success"] is False
        assert result["error"] == "timeout"


def test_predict_connection_error() -> None:
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
        result = predict({"ip": 123})
        assert result["success"] is False
        assert result["error"] == "connection_error"
