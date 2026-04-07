from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (  # noqa: E402
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    OUTPUTS_DIR,
    PREDICTION_THRESHOLD,
    PROCESSED_TRAIN_PATH,
    TARGET_COLUMN,
    ensure_directories,
)
from src.train import RANDOM_STATE  # noqa: E402


def get_validation_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_parquet(PROCESSED_TRAIN_PATH)
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def extract_feature_importance(model: object, feature_names: list[str]) -> pd.Series:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.ravel(estimator.coef_))
    else:
        importances = np.zeros(len(feature_names), dtype=float)

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> Path:
    matrix_path = OUTPUTS_DIR / "confusion_matrix.png"
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Legitimate", "Fraud"]).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=200)
    plt.close(fig)
    return matrix_path


def save_roc_curve(y_true: pd.Series, probabilities: np.ndarray, auc: float) -> Path:
    roc_path = OUTPUTS_DIR / "roc_curve.png"
    fpr, tpr, _ = roc_curve(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}", color="#1f77b4", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)
    return roc_path


def save_feature_importance(importances: pd.Series) -> Path:
    importance_path = OUTPUTS_DIR / "feature_importance.png"
    top_features = importances.head(10).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features.index, top_features.values, color="#2ca02c")
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig(importance_path, dpi=200)
    plt.close(fig)
    return importance_path


def main() -> None:
    ensure_directories()
    metadata = joblib.load(METADATA_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    _, x_valid, _, y_valid = get_validation_split()

    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)

    metrics = {
        "model_name": metadata["champion_model"],
        "auc": float(roc_auc_score(y_valid, probabilities)),
        "log_loss": float(log_loss(y_valid, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_valid, predictions, zero_division=0)),
    }

    save_confusion_matrix(y_valid, predictions)
    save_roc_curve(y_valid, probabilities, metrics["auc"])
    save_feature_importance(extract_feature_importance(model, FEATURE_COLUMNS))

    metrics_path = OUTPUTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved confusion matrix to: {OUTPUTS_DIR / 'confusion_matrix.png'}")
    print(f"Saved ROC curve to: {OUTPUTS_DIR / 'roc_curve.png'}")
    print(f"Saved feature importance to: {OUTPUTS_DIR / 'feature_importance.png'}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
