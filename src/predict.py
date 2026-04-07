from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (  # noqa: E402
    FEATURE_COLUMNS,
    PREDICTION_THRESHOLD,
    build_inference_frame,
    load_artifacts,
)


class FraudPredictor:
    def __init__(self) -> None:
        self.model, self.encoders, self.metadata = load_artifacts()
        self.feature_columns = self.metadata.get("feature_columns", FEATURE_COLUMNS)

    def predict_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        features = build_inference_frame(
            payload=payload,
            encoders=self.encoders,
            feature_columns=self.feature_columns,
        )
        fraud_probability = float(self.model.predict_proba(features)[0, 1])
        prediction = int(fraud_probability >= PREDICTION_THRESHOLD)

        return {
            "fraud_probability": fraud_probability,
            "prediction": prediction,
            "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-record fraud prediction")
    parser.add_argument(
        "--payload",
        required=True,
        help="JSON payload containing ip, app, device, os, channel, and click_time.",
    )
    args = parser.parse_args()

    predictor = FraudPredictor()
    result = predictor.predict_record(json.loads(args.payload))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
