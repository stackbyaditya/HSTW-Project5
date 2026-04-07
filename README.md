# TalkingData Fraud Detection MLOps Pipeline

## Project Title

TalkingData Ad Fraud Detection pipeline with preprocessing, feature engineering, multi-model training, model evaluation, FastAPI inference, testing, Docker packaging, CI automation, and Render-ready deployment settings.

## Problem Statement

Mobile ad platforms face fraudulent click traffic that inflates campaign metrics and wastes marketing spend. This project builds a lightweight end-to-end machine learning pipeline to predict whether a click is fraud-related using a compact sample of the TalkingData dataset so the full workflow can run quickly on modest hardware.

## Dataset Description

The project uses the Kaggle TalkingData AdTracking Fraud Detection dataset. For local development and training, the pipeline intentionally uses `data/train_sample.csv` instead of the full `train.csv` to keep preprocessing and model training fast and practical.

Training columns:

- `ip`
- `app`
- `device`
- `os`
- `channel`
- `click_time`
- `is_attributed`

Additional files in the repository include `data/test_supplement.csv` for unlabeled scoring experiments and `data/sample_submission.csv` as Kaggle output reference material.

## Architecture Overview

1. `src/preprocess.py` loads `data/train_sample.csv`, applies memory-efficient dtypes, extracts datetime features, builds aggregation features, label-encodes categorical columns, and saves processed artifacts.
2. `src/train.py` trains Logistic Regression, Random Forest, XGBoost, and LightGBM on the processed dataset, evaluates them on a stratified validation split, selects a champion model, and stores metadata.
3. `src/evaluate.py` prints a clean comparison table for all trained models.
4. `src/visualize.py` recreates the validation split, scores the champion model, and saves plots plus a metrics JSON file.
5. `src/predict.py` loads the saved model and encoders to score single records.
6. `api/main.py` serves the model through FastAPI with startup-time artifact loading.
7. `tests/test_api.py` validates the health and prediction endpoints.

## Features Engineered

- Time-based features: `hour`, `day`, `weekday`, `minute`
- Aggregation features:
  - `clicks_per_ip`
  - `unique_apps_per_ip`
  - `clicks_per_ip_hour`
  - `clicks_per_channel`
  - `clicks_per_app_os`
- Encoded categorical features:
  - `ip`
  - `app`
  - `device`
  - `os`
  - `channel`

## Models Used

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Class imbalance is handled with `class_weight="balanced"` for sklearn models and `scale_pos_weight` for boosting models.

## Results

Champion model: `XGBoost`

Validation metrics from `outputs/metrics.json`:

- AUC: `0.9757777220969402`
- Log Loss: `0.022477393390349744`
- Accuracy: `0.99345`
- Precision: `0.21333333333333335`
- Recall: `0.7111111111111111`
- F1 Score: `0.3282051282051282`

Artifacts:

- ![Confusion Matrix](outputs/confusion_matrix.png)
- ![ROC Curve](outputs/roc_curve.png)
- ![Feature Importance](outputs/feature_importance.png)

## API Usage

### Health Check

```bash
curl http://127.0.0.1:8000/
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"ip\":87540,\"app\":12,\"device\":1,\"os\":13,\"channel\":497,\"click_time\":\"2017-11-07T09:30:38\"}"
```

Response:

```json
{
  "fraud_probability": 0.10928571224212646,
  "prediction": 0,
  "label": "LEGITIMATE"
}
```

## How to Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the pipeline:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/visualize.py
```

3. Start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

4. Run tests:

```bash
python -m pytest tests/ -v
```

## Docker Instructions

Build the image:

```bash
docker build -t fraud-api .
```

Run the container:

```bash
docker run -p 8000:8000 fraud-api
```

The container starts the API with:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## CI/CD Explanation

The GitHub Actions workflow in `.github/workflows/ci.yml`:

- installs Python dependencies
- preprocesses and trains when `data/train_sample.csv` is present
- generates plots and metrics artifacts
- runs the API tests only when model artifacts are available
- builds the Docker image

This keeps CI resilient when dataset files are intentionally absent from the repository.

## Deployment (Render)

The project is Render-ready through:

- `api/main.py` reading `PORT` with a fallback to `8000`
- `render.yaml` configured for Docker deployment
- the Docker image exposing the FastAPI application

Render start behavior is compatible with:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

## Future Improvements

- add real external test labels for a proper holdout evaluation
- replace fallback aggregation values in inference with online feature store lookups
- add model monitoring and drift alerts
- version training artifacts with a registry
- extend tests to cover missing-artifact and validation-error flows
