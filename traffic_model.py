# traffic_model.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor

CSV_PATH = "hourly_counts_processed_demo.csv"

def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path, parse_dates=["timestamp"])

    df_raw = (
        df_raw.dropna(subset=["timestamp"])
              .sort_values("timestamp")
              .reset_index(drop=True)
    )

    # vehicles_count cleaning (same as your notebook)
    df_raw["vehicles_count"] = (
        pd.to_numeric(df_raw["vehicles_count"], errors="coerce")
          .fillna(0)
          .clip(lower=0)
          .round()
          .astype(int)
    )

    # weather & is_holiday reconstruction (reuse your current code)
    heavy = pd.to_numeric(df_raw["weather_HeavyRain"], errors="coerce").fillna(0).astype(float) > 0
    rain  = pd.to_numeric(df_raw["weather_Rain"],      errors="coerce").fillna(0).astype(float) > 0

    df_raw["weather"] = np.select(
        [heavy,  rain],
        ["HeavyRain", "Rain"],
        default="Clear"
    )

    if "is_holiday" not in df_raw.columns:
        df_raw["is_holiday"] = "No"
        # (you can paste your synthetic holiday logic here if you want)

    df_raw = df_raw[["timestamp", "vehicles_count", "weather", "is_holiday"]]
    return df_raw


def build_feature_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df = pd.get_dummies(df, columns=["weather", "is_holiday"], drop_first=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    for lag in [1, 2, 3, 24, 168]:
        df[f"lag_{lag}"] = df["vehicles_count"].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df


def train_models(df: pd.DataFrame, horizon_days: int = 14):
    feature_cols = [
        "hour", "day_of_week", "is_weekend",
    ] + [c for c in df.columns if c.startswith("weather_") or c.startswith("is_holiday_")] + \
        [c for c in df.columns if c.startswith("lag_")]

    target_col = "vehicles_count"

    cutoff = df["timestamp"].max() - pd.Timedelta(days=horizon_days)
    train = df[df["timestamp"] <= cutoff].copy()
    test  = df[df["timestamp"]  > cutoff].copy()

    X_train, y_train = train[feature_cols], train[target_col]
    X_test,  y_test  = test[feature_cols],  test[target_col]

    # baseline: lag_168
    y_pred_naive = test["lag_168"].values
    mae_naive = mean_absolute_error(y_test, y_pred_naive)

    # Poisson
    poisson = PoissonRegressor(alpha=1e-4, max_iter=300)
    poisson.fit(X_train, y_train)
    pred_poisson = poisson.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    def _mae(y_true, y_hat):
        return mean_absolute_error(y_true, y_hat)

    mae_p = _mae(y_test, pred_poisson)
    mae_rf = _mae(y_test, pred_rf)

    if mae_rf <= mae_p:
        best_model = rf
        y_hat_test = pred_rf
    else:
        best_model = poisson
        y_hat_test = pred_poisson

    return {
        "model": best_model,
        "feature_cols": feature_cols,
        "train_df": train,
        "test_df": test,
        "y_test": y_test,
        "y_hat_test": y_hat_test,
        "mae": {
            "naive": float(mae_naive),
            "poisson": float(mae_p),
            "rf": float(mae_rf),
        }
    }


def build_feature_row(history_df: pd.DataFrame, current_time: pd.Timestamp):
    row = {}
    row["hour"] = current_time.hour
    row["day_of_week"] = current_time.dayofweek
    row["is_weekend"] = 1 if row["day_of_week"] >= 5 else 0

    row["weather_HeavyRain"] = 0
    row["weather_Rain"] = 0
    row["is_holiday_Yes"] = 0

    last_vals = history_df["vehicles_count"].values
    for lag in [1, 2, 3, 24, 168]:
        if len(last_vals) >= lag:
            row[f"lag_{lag}"] = last_vals[-lag]
        else:
            row[f"lag_{lag}"] = last_vals[-1] if len(last_vals) > 0 else 0.0

    return pd.DataFrame([row])

def forecast_next_hours(
    model,
    feature_cols,
    full_df,
    hours: int = 24,
    start_time=None,
) -> pd.DataFrame:
    """
    Forecast `hours` into the future.

    - If start_time is None:
        start right after the last timestamp in the data.
    - If start_time is provided:
        start from that timestamp.
        If it's earlier than the last data timestamp, we clamp up to the last
        timestamp to avoid 'forecasting into the past' using the model.
    """
    history = (
        full_df[["timestamp", "vehicles_count"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    last_time_in_data = history["timestamp"].iloc[-1]

    if start_time is None:
        current_time = last_time_in_data
    else:
        current_time = pd.Timestamp(start_time)
        if current_time < last_time_in_data:
            current_time = last_time_in_data

    future_times = [
        current_time + pd.Timedelta(hours=i) for i in range(1, hours + 1)
    ]
    preds = []

    for t in future_times:
        x_row = build_feature_row(history, t)
        # ensure all features are present
        for col in feature_cols:
            if col not in x_row.columns:
                x_row[col] = 0
        x_row = x_row[feature_cols]
        y_hat = float(model.predict(x_row)[0])

        preds.append({"timestamp": t, "pred_mean": y_hat})

        # append prediction into history so next-hour lags work
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {"timestamp": [t], "vehicles_count": [y_hat]}
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(preds)


def detect_anomalies(y_test, y_hat_test, test_df, threshold: float = 3.0) -> pd.DataFrame:
    resid = y_test.values - y_hat_test
    ts = test_df["timestamp"].values

    def mad(x):
        med = np.median(x)
        return np.median(np.abs(x - med))

    window = min(len(resid), 24*14)
    recent = resid[-window:]
    scale = mad(recent) * 1.4826 if mad(recent) > 0 else (np.std(recent) + 1e-6)
    severity = np.abs(resid) / (scale if scale > 0 else 1e-6)
    flags = severity > threshold

    anom_df = pd.DataFrame({
        "timestamp": ts,
        "actual": y_test.values,
        "pred": y_hat_test,
        "residual": resid,
        "severity": severity,
        "is_anomaly": flags,
    })
    return anom_df
