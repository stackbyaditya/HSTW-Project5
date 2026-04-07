# Streamlit Frontend for Ad Click Fraud Detection

This frontend provides a clean Streamlit interface to collect click features and call the deployed fraud detection API for real-time predictions.

## Features

- Sidebar-based form inputs for click metadata
- Deployed API integration through a dedicated utility module
- Prediction result display with clear fraud/legitimate status
- Probability visualization using percentage, metric, and progress bar
- Input summary table for quick verification
- Graceful handling of timeout, network, HTTP, and invalid-response errors

## Setup and Run

1. Move to frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Streamlit app:

```bash
streamlit run app.py
```

## Deployment on Render

- **Root Directory**: `frontend`
- **Build Command**:

```bash
pip install -r requirements.txt
```

- **Start Command**:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Notes:

- The backend API is already deployed; this frontend only consumes it.
- On Render, the first request can be slow due to cold start / waking up.

## API Information

- Base URL: `https://fraud-detection-api-wb1m.onrender.com`
- Endpoint: `POST /predict`
- Full endpoint URL: `https://fraud-detection-api-wb1m.onrender.com/predict`
- API docs: `https://fraud-detection-api-wb1m.onrender.com/docs`

Expected request payload format:

```json
{
  "ip": 87540,
  "app": 12,
  "device": 1,
  "os": 13,
  "channel": 497,
  "click_time": "2017-11-07 09:30:38"
}
```

## Cold Start Note

This service is deployed on Render. If the app has been idle, the first request can time out while the backend wakes up. Retry after a few seconds if you see a timeout message.
