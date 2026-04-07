from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionRequest(BaseModel):
    ip: int
    app: int
    device: int
    os: int
    channel: int
    click_time: datetime


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., ge=0, le=1)
    label: Literal["FRAUD", "LEGITIMATE"]
