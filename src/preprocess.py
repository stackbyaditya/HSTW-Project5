from __future__ import annotations

import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (  # noqa: E402
    ENCODERS_PATH,
    FEATURE_COLUMNS,
    PROCESSED_TRAIN_PATH,
    TARGET_COLUMN,
    add_aggregation_features,
    add_datetime_features,
    fit_label_encoders,
    load_raw_training_data,
)


def main() -> None:
    dataset = load_raw_training_data()
    dataset = add_datetime_features(dataset)
    dataset = add_aggregation_features(dataset)
    dataset, encoders = fit_label_encoders(dataset)
    processed = dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    processed.to_parquet(PROCESSED_TRAIN_PATH, index=False)
    joblib.dump(encoders, ENCODERS_PATH)

    print(f"Loaded rows: {len(dataset):,}")
    print(f"Processed dataset saved to: {PROCESSED_TRAIN_PATH}")
    print(f"Encoders saved to: {ENCODERS_PATH}")


if __name__ == "__main__":
    main()
