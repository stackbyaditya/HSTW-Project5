"""Streamlit frontend for deployed ad-click fraud detection API."""

from __future__ import annotations

import os
import hashlib
import ipaddress
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from utils import predict

APP_MAP = {
    101: "Chrome",
    202: "Safari",
    303: "Firefox",
}

OS_MAP = {
    13: "Windows",
    17: "Android",
    19: "iOS",
}

DEVICE_MAP = {
    1: "Mobile",
    2: "Desktop",
}

CHANNEL_MAP = {
    497: "Google Ads",
    130: "Facebook Ads",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #1f2937 0%, #0f172a 55%, #020617 100%);
            color: #e2e8f0;
        }
        .hero-card, .glass-card, .result-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.30);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.35);
            margin-bottom: 1rem;
        }
        .summary-card {
            padding: 0.7rem 0.9rem 0.25rem 0.9rem;
        }
        .hero-title {
            text-align: center;
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 45%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-subtitle {
            text-align: center;
            color: #cbd5e1;
            margin-bottom: 0;
        }
        .ad-shell {
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.35);
            margin-bottom: 0.65rem;
        }
        .ad-note {
            color: #cbd5e1;
            font-size: 0.93rem;
            margin-top: 0.2rem;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 12px;
            border: 1px solid rgba(147, 197, 253, 0.60);
            font-weight: 700;
        }
        hr.premium-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, rgba(148,163,184,0.1), rgba(148,163,184,0.8), rgba(148,163,184,0.1));
            margin: 1rem 0 1.2rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _browser_to_app_code(user_agent: str) -> tuple[int, str]:
    normalized = user_agent.lower()
    mapping = [
        ("edg", 101, "Chrome"),
        ("chrome", 101, "Chrome"),
        ("firefox", 303, "Firefox"),
        ("safari", 202, "Safari"),
        ("opera", 303, "Firefox"),
    ]
    for token, code, label in mapping:
        if token in normalized:
            return code, label
    return 100, "Unknown Browser"


def _detect_ip_as_int() -> tuple[int, str]:
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        response.raise_for_status()
        ip_text = str(response.json().get("ip", "")).strip()
        if not ip_text:
            raise ValueError("Missing IP in response")
        ip_object = ipaddress.ip_address(ip_text)
        if ip_object.version == 4:
            return int(ip_object), f"{ip_text} (auto-detected)"
        hashed_value = int(hashlib.sha256(ip_text.encode("utf-8")).hexdigest()[:12], 16)
        return hashed_value, f"{ip_text} (auto-detected, hashed)"
    except Exception:
        fallback_ip = 87540
        return fallback_ip, f"{fallback_ip} (default fallback)"


def _extract_user_agent() -> str:
    try:
        headers = dict(st.context.headers) if st.context and st.context.headers else {}
    except Exception:
        headers = {}
    return str(headers.get("User-Agent", "")).strip()


def _initialize_state() -> None:
    if "ad_clicked" not in st.session_state:
        st.session_state.ad_clicked = False
    if "click_timestamp" not in st.session_state:
        st.session_state.click_timestamp = None
    if "formatted_click_date" not in st.session_state:
        st.session_state.formatted_click_date = "-"
    if "formatted_click_time" not in st.session_state:
        st.session_state.formatted_click_time = "-"
    if "ip_value" not in st.session_state:
        ip_value, ip_label = _detect_ip_as_int()
        st.session_state.ip_value = ip_value
        st.session_state.ip_label = ip_label
    if "app_value" not in st.session_state:
        user_agent = _extract_user_agent()
        app_value, app_label = _browser_to_app_code(user_agent)
        if not user_agent:
            app_label = f"{app_label} (default fallback)"
        else:
            app_label = f"{app_label} (auto-detected)"
        st.session_state.app_value = app_value
        st.session_state.app_label = app_label


def _parse_prediction_response(response_data: dict[str, Any]) -> tuple[bool, float | None, str]:
    raw_prediction = (
        response_data.get("prediction")
        if response_data.get("prediction") is not None
        else response_data.get("predicted_class")
    )
    raw_label = response_data.get("label")
    raw_probability = (
        response_data.get("probability")
        if response_data.get("probability") is not None
        else response_data.get("fraud_probability")
    )
    model_name = str(response_data.get("model") or "XGBoost")

    is_fraud = False
    if isinstance(raw_label, str):
        is_fraud = raw_label.strip().lower() in {"fraud", "fraudulent", "bot", "1"}
    elif raw_prediction is not None:
        try:
            is_fraud = int(raw_prediction) == 1
        except (TypeError, ValueError):
            is_fraud = str(raw_prediction).strip().lower() in {"fraud", "fraudulent", "bot", "1"}

    probability_value = None
    if raw_probability is not None:
        try:
            probability_value = float(raw_probability)
            if probability_value > 1:
                probability_value = probability_value / 100.0
            probability_value = max(0.0, min(1.0, probability_value))
        except (TypeError, ValueError):
            probability_value = None

    return is_fraud, probability_value, model_name


st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

_inject_styles()
_initialize_state()

st.caption("Frontend running successfully")

st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">🚨 Ad Click Fraud Detection System</h1>
        <p class="hero-subtitle">Detect whether a click is Human or Bot using ML</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<hr class='premium-divider'>", unsafe_allow_html=True)
st.caption("Use the sidebar for optional feature controls (Device, OS, Channel).")

st.sidebar.header("Optional Feature Controls")
device_value = st.sidebar.selectbox(
    "Device",
    options=list(DEVICE_MAP.keys()),
    index=0,
    format_func=lambda x: DEVICE_MAP.get(x, f"Unknown ({x})"),
)
os_value = st.sidebar.selectbox(
    "Operating System",
    options=list(OS_MAP.keys()),
    index=0,
    format_func=lambda x: OS_MAP.get(x, f"Unknown ({x})"),
)
channel_value = st.sidebar.selectbox(
    "Ad Channel",
    options=list(CHANNEL_MAP.keys()),
    index=0,
    format_func=lambda x: CHANNEL_MAP.get(x, f"Unknown ({x})"),
)
st.sidebar.markdown("---")
st.sidebar.subheader("About Project")
st.sidebar.info(
    "This app sends click data to a deployed ML model and predicts fraud in real-time."
)

st.markdown(
    """
    <div class="glass-card">
        <h4 style="margin-top:0;">Sponsored Ad</h4>
        <p style="margin-bottom:0.2rem;">Interact with this ad to generate a real click event timestamp.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="ad-shell">', unsafe_allow_html=True)
ad_image_path = Path("assets") / "ad_banner.png"
try:
    if ad_image_path.exists():
        # Backward-compatible across Streamlit versions (older versions don't support use_container_width)
        st.image(str(ad_image_path), use_column_width=True)
    else:
        st.info("Ad banner image not found. Click the CTA below to simulate a click event.")
except Exception:
    st.info("Ad banner image could not be loaded. Click the CTA below to simulate a click event.")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    "<p class='ad-note'>Sponsored placement preview. Click the CTA below to simulate a click event.</p>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🖱️ Click Ad - Learn More", use_container_width=True):
    current_ts = datetime.now()
    st.session_state.ad_clicked = True
    st.session_state.click_timestamp = current_ts
    st.session_state.formatted_click_date = current_ts.strftime("%Y-%m-%d")
    st.session_state.formatted_click_time = current_ts.strftime("%H:%M:%S")
    st.success("Click event captured successfully.")

if not st.session_state.ad_clicked:
    st.warning("Generate a click event first by tapping the ad card above.")

summary_payload = {
    "IP (Auto-detected)": st.session_state.ip_label,
    "Browser/App": APP_MAP.get(int(st.session_state.app_value), st.session_state.app_label),
    "Device": DEVICE_MAP.get(int(device_value), f"Unknown ({int(device_value)})"),
    "OS": OS_MAP.get(int(os_value), f"Unknown ({int(os_value)})"),
    "Channel": CHANNEL_MAP.get(int(channel_value), f"Unknown ({int(channel_value)})"),
    "click_date": st.session_state.formatted_click_date,
    "click_time": st.session_state.formatted_click_time,
}

st.markdown(
    """
    <div class="glass-card summary-card">
        <h4 style="margin-top:0;">Session Summary (Read-Only)</h4>
    </div>
    """,
    unsafe_allow_html=True,
)
st.dataframe(pd.DataFrame([summary_payload]), use_container_width=True, hide_index=True)
st.caption("Encoded feature values are preserved for model payload compatibility.")
st.markdown("<br>", unsafe_allow_html=True)

predict_clicked = st.button("🔍 Predict Fraud", type="primary", use_container_width=True)

if predict_clicked:
    if not st.session_state.ad_clicked or not st.session_state.click_timestamp:
        st.error("Please click the ad visual first to capture a valid click timestamp.")
    else:
        numeric_fields = {
            "ip": st.session_state.ip_value,
            "app": st.session_state.app_value,
            "device": device_value,
            "os": os_value,
            "channel": channel_value,
        }
        if any(value is None for value in numeric_fields.values()):
            st.warning("Missing one or more required values. Refresh and try again.")
        elif any(int(value) < 0 for value in numeric_fields.values()):
            st.error("Negative values are not allowed.")
        else:
            payload = {
                "ip": int(st.session_state.ip_value),
                "app": int(st.session_state.app_value),
                "device": int(device_value),
                "os": int(os_value),
                "channel": int(channel_value),
                "click_time": st.session_state.click_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }

            with st.spinner("⏳ Sending request to cloud model..."):
                result = predict(payload)

            if not result["success"]:
                error_msg = str(result.get("error") or "").lower()
                if "timeout" in error_msg:
                    st.error("Model is waking up (cold start), please retry.")
                    st.info("Retry in a few seconds if the service was idle.")
                else:
                    st.error("Service temporarily unavailable")
                    if error_msg:
                        st.caption(f"Details: {error_msg}")
            else:
                response_data: dict[str, Any] = result.get("data") or {}
                is_fraud, probability_value, model_name = _parse_prediction_response(response_data)

                st.markdown(
                    """
                    <div class="result-card">
                        <h4 style="margin-top:0;">Prediction Result</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if is_fraud:
                    st.error("🔴 Fraudulent Click")
                else:
                    st.success("🟢 Legitimate Click")

                if probability_value is not None:
                    probability_percent = probability_value * 100
                    st.metric("Fraud Probability", f"{probability_percent:.2f}%")
                    st.progress(probability_value)

                st.info(f"Model used: {model_name}")
