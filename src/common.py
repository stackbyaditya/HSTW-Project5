from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_TRAIN_PATH = DATA_DIR / "train_sample.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DIR / "train_processed.parquet"
ENCODERS_PATH = MODELS_DIR / "encoders.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "metadata.joblib"

TARGET_COLUMN = "is_attributed"
UNKNOWN_TOKEN = "__UNKNOWN__"
PREDICTION_THRESHOLD = 0.5

SOURCE_COLUMNS = ["ip", "app", "device", "os", "channel", "click_time", TARGET_COLUMN]
CATEGORICAL_COLUMNS = ["ip", "app", "device", "os", "channel"]
TIME_FEATURE_COLUMNS = ["hour", "day", "weekday", "minute"]
AGGREGATION_FEATURE_COLUMNS = [
    "clicks_per_ip",
    "unique_apps_per_ip",
    "clicks_per_ip_hour",
    "clicks_per_channel",
    "clicks_per_app_os",
]
FEATURE_COLUMNS = (
    CATEGORICAL_COLUMNS + TIME_FEATURE_COLUMNS + AGGREGATION_FEATURE_COLUMNS
)

OPTIMIZED_DTYPES = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
    TARGET_COLUMN: "uint8",
}


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_training_data() -> pd.DataFrame:
    ensure_directories()
    return pd.read_csv(
        RAW_TRAIN_PATH,
        usecols=SOURCE_COLUMNS,
        dtype=OPTIMIZED_DTYPES,
        parse_dates=["click_time"],
    )


def add_datetime_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["hour"] = enriched["click_time"].dt.hour.astype("uint8")
    enriched["day"] = enriched["click_time"].dt.day.astype("uint8")
    enriched["weekday"] = enriched["click_time"].dt.weekday.astype("uint8")
    enriched["minute"] = enriched["click_time"].dt.minute.astype("uint8")
    return enriched


def add_aggregation_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["clicks_per_ip"] = (
        enriched.groupby("ip")["ip"].transform("size").astype("uint32")
    )
    enriched["unique_apps_per_ip"] = (
        enriched.groupby("ip")["app"].transform("nunique").astype("uint16")
    )
    enriched["clicks_per_ip_hour"] = (
        enriched.groupby(["ip", "hour"])["ip"].transform("size").astype("uint16")
    )
    enriched["clicks_per_channel"] = (
        enriched.groupby("channel")["channel"].transform("size").astype("uint32")
    )
    enriched["clicks_per_app_os"] = (
        enriched.groupby(["app", "os"])["app"].transform("size").astype("uint32")
    )
    return enriched


def fit_label_encoders(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoded = frame.copy()
    encoders: dict[str, LabelEncoder] = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        values = encoded[column].astype(str)
        encoder.fit(values)

        if UNKNOWN_TOKEN not in encoder.classes_:
            encoder.classes_ = np.unique(np.append(encoder.classes_, UNKNOWN_TOKEN))

        encoded[column] = encoder.transform(values).astype("int32")
        encoders[column] = encoder

    return encoded, encoders


def encode_with_unknown(values: pd.Series, encoder: LabelEncoder) -> pd.Series:
    lookup = {label: index for index, label in enumerate(encoder.classes_)}
    unknown_index = lookup.get(UNKNOWN_TOKEN, -1)
    encoded = values.astype(str).map(lookup).fillna(unknown_index)
    return encoded.astype("int32")


def build_inference_frame(
    payload: dict,
    encoders: dict[str, LabelEncoder],
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = feature_columns or FEATURE_COLUMNS
    frame = pd.DataFrame([payload]).copy()
    frame["click_time"] = pd.to_datetime(frame["click_time"])

    for column in CATEGORICAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")

    frame = add_datetime_features(frame)

    for column in AGGREGATION_FEATURE_COLUMNS:
        frame[column] = 1

    for column in CATEGORICAL_COLUMNS:
        frame[column] = encode_with_unknown(frame[column], encoders[column])

    return frame[columns]


def load_artifacts() -> tuple[object, dict[str, LabelEncoder], dict]:
    model = joblib.load(BEST_MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, encoders, metadata
