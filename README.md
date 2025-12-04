Traffic Forecast Model

This project contains a small, self-contained machine learning service for forecasting hourly traffic volume and detecting anomalies. It was originally built as a JupyterLab notebook and then refactored into a reusable Python module and a FastAPI web service. A simple HTML/Chart.js dashboard is included to visualize the forecasts.

What This Model Does

The model predicts future hourly traffic counts based on historical patterns. It uses:

Calendar features
(hour of day, day of week, weekend flag)

Weather categories
(Clear, Rain, HeavyRain)

Holiday effects

Lag features
(1, 2, 3, 24, and 168 hours)

Count-data modeling with Poisson Regression

Non-linear modeling with Random Forest

A baseline forecast using lag-168 (same hour last week)

Performance is evaluated with MAE and MAPE, and the forecast horizon is configurable (24–168 hours).

Anomaly Detection

Residuals between actual and predicted values are scored using median absolute deviation (MAD). Points with high severity (MAD > threshold) are flagged as anomalies. This provides a simple operational monitoring example built on top of the model.

API Endpoints

The FastAPI service exposes:

/api/forecast
Returns a future forecast. Example:
/api/forecast?hours=24&start=2025-12-04T00:00:00

/api/anomalies
Returns anomaly scores for the test window.

/api/model_code
Returns the model source code (traffic_model.py) for transparency.

/
Serves an interactive HTML dashboard with charts.

Project Structure
traffic-forecast-model/
    app.py                – FastAPI service + routes
    traffic_model.py      – ML forecasting and anomaly detection logic
    requirements.txt
    hourly_counts_processed_demo.csv   – Example dataset
    static/
        index.html        – Simple dashboard using Chart.js

How to Run Locally

Install dependencies:

pip install -r requirements.txt


Start the service:

uvicorn app:app --reload --port 8000


Open the dashboard:

http://localhost:8000/

Deployment

The app can be deployed as a single Python web service.
For platforms like Render, use the following start command:

uvicorn app:app --host 0.0.0.0 --port $PORT

Purpose

This project demonstrates how a notebook experiment can be turned into a lightweight, production-style ML forecasting service with:

A clearly defined preprocessing and feature pipeline

Deterministic model logic

Clean API endpoints for inference

A minimal UI for visualizing results

It is intentionally simple and easy to read, focusing on clarity of the ML model rather than infrastructure.
