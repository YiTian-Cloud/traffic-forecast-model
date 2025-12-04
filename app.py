# app.py
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

import pandas as pd

from traffic_model import (
    load_data,
    build_feature_frame,
    train_models,
    forecast_next_hours,
    detect_anomalies,
)

app = FastAPI(title="Traffic Forecast & Anomaly Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for demo; tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the single-page frontend
@app.get("/")
def root():
    return FileResponse("static/index.html")

# Serve any other static assets under /static (if you add CSS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---- Train model on startup ----
df_raw = load_data()
df_features = build_feature_frame(df_raw)
train_info = train_models(df_features, horizon_days=14)

model = train_info["model"]
feature_cols = train_info["feature_cols"]
test_df = train_info["test_df"]
y_test = train_info["y_test"]
y_hat_test = train_info["y_hat_test"]
mae_scores = train_info["mae"]


class ForecastResponseItem(BaseModel):
    timestamp: str
    pred_mean: float


class AnomalyResponseItem(BaseModel):
    timestamp: str
    actual: float
    pred: float
    residual: float
    severity: float
    is_anomaly: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return {"mae": mae_scores}


@app.get("/api/forecast", response_model=List[ForecastResponseItem])
def forecast(hours: int = 24, start: Optional[str] = None):
    """
    Forecast traffic.

    - hours: number of hours to forecast.
    - start: optional ISO datetime string (e.g. '2026-01-10T00:00:00').
             If omitted, we start right after the last timestamp in the data.
    """
    future_df = forecast_next_hours(
        model,
        feature_cols,
        df_features,
        hours=hours,
        start_time=start,
    )
    return [
        ForecastResponseItem(
            timestamp=row["timestamp"].isoformat(),
            pred_mean=row["pred_mean"],
        )
        for _, row in future_df.iterrows()
    ]
    future_df = forecast_next_hours(model, feature_cols, df_features, hours=hours)
    return [
        ForecastResponseItem(
            timestamp=row["timestamp"].isoformat(),
            pred_mean=row["pred_mean"],
        )
        for _, row in future_df.iterrows()
    ]


@app.get("/api/anomalies", response_model=List[AnomalyResponseItem])
def anomalies(threshold: float = 3.0):
    anom_df = detect_anomalies(y_test, y_hat_test, test_df, threshold)
    return [
        AnomalyResponseItem(
            timestamp=row["timestamp"].isoformat(),
            actual=float(row["actual"]),
            pred=float(row["pred"]),
            residual=float(row["residual"]),
            severity=float(row["severity"]),
            is_anomaly=bool(row["is_anomaly"]),
        )
        for _, row in anom_df.iterrows()
        if row["is_anomaly"]
    ]

@app.get("/api/model_code")
def model_code():
    """
    Return the contents of traffic_model.py as plain text.
    """
    code_path = Path(__file__).parent / "traffic_model.py"
    if not code_path.exists():
        return {"code": "# traffic_model.py not found"}
    code = code_path.read_text(encoding="utf-8")
    return {"code": code}
