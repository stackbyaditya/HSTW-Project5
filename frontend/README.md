# Streamlit Frontend for Ad Click Fraud Detection

This frontend provides an interactive Streamlit user interface to capture click features, auto-detect client metadata (IP address and User-Agent browser tokens), and call the deployed FastAPI fraud detection API for real-time inference.

## Features

- Dynamic hero banner and custom dark-theme styling
- Interactive sponsored ad click simulator capturing click timestamps
- Automatic client IP detection (`api.ipify.org` fallback) and browser user-agent mapping
- Sidebar feature controls for optional attributes (Device, Operating System, Ad Channel)
- Real-time API communication with graceful error handling (cold start timeouts, connection errors, HTTP status errors)
- Probability metric display and progress bar visualization

## Setup and Run

1. Install dependencies from workspace root:

```bash
pip install -r frontend/requirements.txt
```

2. Start the Streamlit app:

```bash
streamlit run frontend/app.py
```

Or from inside the `frontend/` directory:

```bash
cd frontend
streamlit run app.py
```

## Environment Variables

| Variable | Description | Default Value |
|---|---|---|
| `FRAUD_API_URL` | Base URL of the backend API | `https://fraud-detection-api-wb1m.onrender.com` |

To connect the frontend to a local backend API during development:

```bash
FRAUD_API_URL=http://127.0.0.1:8000 streamlit run frontend/app.py
```

## Deployment Options

### 1. Streamlit Community Cloud (Recommended)

- **Repository**: Select `stackbyaditya/HSTW-Project5`
- **Branch**: `main`
- **Main file path**: `frontend/app.py`
- **Secrets / Environment Variables**:
  - `FRAUD_API_URL`: `https://fraud-detection-api-wb1m.onrender.com`

### 2. Render Deployment

- **Root Directory**: `.`
- **Build Command**: `pip install -r frontend/requirements.txt`
- **Start Command**: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`

### 3. Vercel Note

Streamlit requires continuous long-running WebSocket connections and cannot run as a stateless Vercel Serverless Function. `.vercelignore` and `vercel.json` are included in the repository root so Vercel ignores `frontend/app.py` and avoids build failures. Host the Streamlit frontend on Streamlit Community Cloud or Render Web Services.
