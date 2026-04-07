from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import METADATA_PATH  # noqa: E402


def main() -> None:
    metadata = joblib.load(METADATA_PATH)
    results = pd.DataFrame(metadata["results"])[["Model", "AUC", "Log Loss", "Status"]]

    print(results.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print()
    print(f"Champion model: {metadata['champion_model']}")
    print(f"Dataset size used: {metadata['dataset_size']:,}")
    print(f"Fraud rate: {metadata['fraud_rate']:.5%}")


if __name__ == "__main__":
    main()
