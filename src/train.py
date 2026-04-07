from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (  # noqa: E402
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    PROCESSED_TRAIN_PATH,
    TARGET_COLUMN,
)

RANDOM_STATE = 42


def build_models(scale_pos_weight: float) -> dict[str, object]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=400,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
    }


def main() -> None:
    dataset = pd.read_parquet(PROCESSED_TRAIN_PATH)
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / max(positive_count, 1)

    trained_models: dict[str, object] = {}
    results: list[dict[str, object]] = []

    for model_name, model in build_models(scale_pos_weight).items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        auc = roc_auc_score(y_valid, probabilities)
        loss = log_loss(y_valid, probabilities, labels=[0, 1])

        trained_models[model_name] = model
        results.append(
            {
                "Model": model_name,
                "AUC": float(auc),
                "Log Loss": float(loss),
            }
        )
        print(f"Trained {model_name}: AUC={auc:.5f}, Log Loss={loss:.5f}")

    champion = max(results, key=lambda item: (item["AUC"], -item["Log Loss"]))

    for result in results:
        result["Status"] = "Champion" if result["Model"] == champion["Model"] else "Challenger"

    joblib.dump(trained_models[champion["Model"]], BEST_MODEL_PATH)
    metadata = {
        "champion_model": champion["Model"],
        "dataset_size": int(len(dataset)),
        "fraud_rate": float(target.mean()),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "validation_size": int(len(x_valid)),
        "results": results,
    }
    joblib.dump(metadata, METADATA_PATH)

    print(f"Champion model saved to: {BEST_MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
