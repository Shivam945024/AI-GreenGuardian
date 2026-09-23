import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "pollution_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

POLLUTION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "pollution_model.pkl"
)

RISK_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "risk_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity",
    "Wind Speed"
]


# ============================================================
# LOAD FILES
# ============================================================

print("\n========================================")
print(" AI-GreenGuardian Model Evaluation")
print("========================================\n")


for path in [
    DATA_PATH,
    POLLUTION_MODEL_PATH,
    RISK_MODEL_PATH,
    SCALER_PATH
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


df = pd.read_csv(
    DATA_PATH
)


pollution_model = joblib.load(
    POLLUTION_MODEL_PATH
)

risk_model = joblib.load(
    RISK_MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# CLEAN DATA
# ============================================================

required_columns = FEATURES + [
    "AQI"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing)
    )


df = df[
    required_columns
].copy()


for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna()


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df[
    FEATURES
]

y_aqi = df[
    "AQI"
]


# ============================================================
# SCALE
# ============================================================

X_scaled = scaler.transform(
    X
)


# ============================================================
# POLLUTION MODEL
# ============================================================

print(
    "\nEvaluating pollution model..."
)


aqi_prediction = pollution_model.predict(
    X_scaled
)


aqi_mae = mean_absolute_error(
    y_aqi,
    aqi_prediction
)

aqi_mse = mean_squared_error(
    y_aqi,
    aqi_prediction
)

aqi_rmse = aqi_mse ** 0.5

aqi_r2 = r2_score(
    y_aqi,
    aqi_prediction
)


# ============================================================
# RISK TARGET
# ============================================================

def calculate_risk(row):

    pm25 = row["PM2.5"]
    pm10 = row["PM10"]
    no2 = row["NO2"]
    so2 = row["SO2"]
    co = row["CO"]
    o3 = row["O3"]

    temperature = row["Temperature"]
    humidity = row["Humidity"]
    wind = row["Wind Speed"]

    pm25_score = min(
        100,
        pm25 / 2
    )

    pm10_score = min(
        100,
        pm10 / 3
    )

    no2_score = min(
        100,
        no2 / 2
    )

    so2_score = min(
        100,
        so2 / 1.5
    )

    co_score = min(
        100,
        co / 0.05
    )

    o3_score = min(
        100,
        o3 / 1.5
    )

    temperature_score = min(
        100,
        max(
            0,
            (temperature - 20) * 3
        )
    )

    humidity_score = min(
        100,
        max(
            0,
            (humidity - 60) * 2
        )
    )

    wind_score = max(
        0,
        100 - wind * 10
    )

    risk = (
        pm25_score * 0.25
        + pm10_score * 0.12
        + no2_score * 0.12
        + so2_score * 0.05
        + co_score * 0.05
        + o3_score * 0.05
        + temperature_score * 0.05
        + humidity_score * 0.06
        + wind_score * 0.25
    )

    return max(
        0,
        min(
            100,
            risk
        )
    )


y_risk = df.apply(
    calculate_risk,
    axis=1
)


# ============================================================
# RISK MODEL
# ============================================================

print(
    "Evaluating risk model..."
)


risk_prediction = risk_model.predict(
    X_scaled
)


risk_mae = mean_absolute_error(
    y_risk,
    risk_prediction
)

risk_mse = mean_squared_error(
    y_risk,
    risk_prediction
)

risk_rmse = risk_mse ** 0.5

risk_r2 = r2_score(
    y_risk,
    risk_prediction
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print(" POLLUTION MODEL")
print("========================================")

print(
    f"MAE  : {aqi_mae:.2f}"
)

print(
    f"RMSE : {aqi_rmse:.2f}"
)

print(
    f"R²   : {aqi_r2:.4f}"
)


print("\n========================================")
print(" RISK MODEL")
print("========================================")

print(
    f"MAE  : {risk_mae:.2f}"
)

print(
    f"RMSE : {risk_rmse:.2f}"
)

print(
    f"R²   : {risk_r2:.4f}"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual AQI": y_aqi.values,
    "Predicted AQI": np.round(
        aqi_prediction,
        2
    ),
    "Actual Risk": np.round(
        y_risk.values,
        2
    ),
    "Predicted Risk": np.round(
        risk_prediction,
        2
    )
})


print("\n========================================")
print(" SAMPLE PREDICTIONS")
print("========================================")

print(
    results.head(10).to_string(
        index=False
    )
)


print(
    "\nModel evaluation completed successfully."
)
