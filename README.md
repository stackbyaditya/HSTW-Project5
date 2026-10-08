# TalkingData Fraud Detection MLOps Pipeline

## Project Title

TalkingData Ad Fraud Detection pipeline with preprocessing, feature engineering, multi-model training, model evaluation, FastAPI inference, Streamlit interactive frontend, automated pytest suite, Docker packaging, CI automation, and Render/Streamlit Cloud/Vercel deployment configurations.

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

```text
User / Browser
      │
      ▼
Streamlit Frontend (Host: Streamlit Community Cloud / Render / Railway)
      │
      ▼ HTTPS POST /predict (Configurable via FRAUD_API_URL)
FastAPI Backend API (Host: Render Docker / Vercel Serverless / Container)
      │
      ▼
ML Model & Pipeline (XGBoost Champion Model)
```

1. **Preprocessing (`src/preprocess.py`)**: Loads `data/train_sample.csv`, applies memory-efficient dtypes, extracts datetime features (`hour`, `day`, `weekday`, `minute`), builds aggregation features (`clicks_per_ip`, `clicks_per_channel`, etc.), label-encodes categorical columns, and saves processed artifacts.
2. **Model Training (`src/train.py`)**: Trains Logistic Regression, Random Forest, XGBoost, and LightGBM models on the processed dataset, evaluates them on a stratified validation split, selects the champion model, and stores model binaries + metadata in `models/`.
3. **Model Evaluation (`src/evaluate.py`)**: Displays comparison metrics across all trained models.
4. **Visualization (`src/visualize.py`)**: Scores the champion model on validation data and generates confusion matrix, ROC curve, and feature importance plots in `outputs/`.
5. **Inference (`src/predict.py`)**: Scores individual click payloads using loaded model artifacts.
6. **FastAPI Backend (`api/main.py`)**: Serves prediction endpoints (`/` health check and `/predict`) with CORS support and lifespan model loading.
7. **Streamlit Frontend (`frontend/app.py`)**: Renders an interactive web application allowing users to simulate click events, auto-detect client metadata (IP/User-Agent), customize input features, and view real-time fraud predictions.
8. **Automated Testing (`tests/`)**: Pytest suite validating backend health/prediction endpoints and frontend utility error-handling mechanisms.

## Deployment Architecture Strategy

The application consists of two decoupled components:

1. **FastAPI Backend**:
   - Long-running REST service deployed via Docker container on **Render** (or Vercel Serverless Function via `api/main.py`).
   - Exposes `/` and `/predict` endpoints with CORS enabled.
2. **Streamlit Frontend**:
   - Reactive WebSocket application built with Streamlit.
   - Deployable on **Streamlit Community Cloud**, **Render Web Service**, **Hugging Face Spaces**, or **Railway**.
   - Note: Streamlit applications require persistent WebSocket connections and cannot run as stateless serverless HTTP functions. `.vercelignore` and `vercel.json` are included to ensure Vercel does not misidentify `frontend/app.py` as a serverless function when linking the repository.

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

Class imbalance is handled with `class_weight="balanced"` for scikit-learn models and `scale_pos_weight` for boosting models.

## Results

Champion model: `XGBoost`

Validation metrics from `outputs/metrics.json`:

- AUC: `0.975777`
- Log Loss: `0.022477`
- Accuracy: `0.99345`
- Precision: `0.213333`
- Recall: `0.711111`
- F1 Score: `0.328205`

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

2. Run the pipeline (optional, pre-built artifacts exist in `models/`):

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/visualize.py
```

3. Start the FastAPI backend API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

4. Start the Streamlit frontend app (in a second terminal):

```bash
streamlit run frontend/app.py
```

5. Run test suite:

```bash
python -m pytest tests/ -v
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PORT` | Port for FastAPI backend service | `8000` |
| `FRAUD_API_URL` | Base URL of the backend API for Streamlit frontend | `https://fraud-detection-api-wb1m.onrender.com` |

## Docker Instructions

Build the backend image:

```bash
docker build -t fraud-api .
```

Run the container:

```bash
docker run -p 8000:8000 -e PORT=8000 fraud-api
```

## CI/CD Explanation

The GitHub Actions workflow in `.github/workflows/ci.yml`:

- installs Python dependencies
- preprocesses and trains when `data/train_sample.csv` is present
- generates plots and metrics artifacts
- runs the API & frontend unit test suite
- builds the Docker container image

## Deployment Instructions

### Option 1: Deploy Frontend to Streamlit Community Cloud (Recommended)
1. Fork or connect this GitHub repository to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Set Main file path to: `frontend/app.py`.
3. Add Environment Variable:
   - `FRAUD_API_URL`: `https://fraud-detection-api-wb1m.onrender.com` (or your deployed backend URL).

### Option 2: Deploy Both Backend and Frontend to Render
`render.yaml` contains pre-configured web service definitions for both components:
- `fraud-api`: Docker web service for FastAPI backend.
- `fraud-frontend`: Native Python web service running `streamlit run frontend/app.py`.

### Option 3: Deploy Backend to Render / Vercel Serverless
- Backend FastAPI app is configured in `render.yaml` (Render Docker) and `vercel.json` (Vercel Python Serverless Function targeting `api/main.py`).
- `.vercelignore` ignores `frontend/` so Vercel does not misidentify `frontend/app.py` as a serverless HTTP endpoint.
