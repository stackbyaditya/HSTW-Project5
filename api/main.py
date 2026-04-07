from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import HealthResponse, PredictionRequest, PredictionResponse  # noqa: E402
from src.predict import FraudPredictor  # noqa: E402

PORT = int(os.environ.get("PORT", 8000))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = FraudPredictor()
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    prediction = request.app.state.predictor.predict_record(payload.model_dump())
    return PredictionResponse(**prediction)


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=PORT)
