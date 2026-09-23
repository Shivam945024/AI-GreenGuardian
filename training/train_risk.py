import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATASET_PATH = os.path.join(
    DATA_DIR,
    "pollution_dataset.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "risk_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
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
# RISK SCORE FUNCTION
# ============================================================

def calculate_training_risk(row):

    pm25 = float(row["PM2.5"])
    pm10 = float(row["PM10"])
    no2 = float(row["NO2"])
    so2 = float(row["SO2"])
    co = float(row["CO"])
    o3 = float(row["O3"])

    temperature = float(
        row["Temperature"]
    )

    humidity = float(
        row["Humidity"]
    )

    wind = float(
        row["Wind Speed"]
    )

    # Pollution components

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

    # Weather components

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

    # Low wind = higher pollution accumulation

    wind_score = max(
        0,
        100 - wind * 10
    )

    # Weighted environmental risk

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

    return round(
        max(
            0,
            min(100, risk)
        ),
        2
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print(" AI-GreenGuardian Risk Model")
print("========================================\n")


if not os.path.exists(
    DATASET_PATH
):

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )


df = pd.read_csv(
    DATASET_PATH
)


# ============================================================
# VALIDATE DATA
# ============================================================

missing = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing)
    )


df = df[
    FEATURES
].copy()


# ============================================================
# CLEAN DATA
# ============================================================

for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna()


for column in FEATURES:

    df = df[
        df[column] >= 0
    ]


if len(df) < 30:

    raise ValueError(
        "At least 30 valid rows are required."
    )


# ============================================================
# CREATE RISK TARGET
# ============================================================

print(
    "\nGenerating environmental risk scores..."
)


df["Risk Score"] = df.apply(
    calculate_training_risk,
    axis=1
)


# ============================================================
# PREPARE DATA
# ============================================================

X = df[
    FEATURES
]

y = df[
    "Risk Score"
]


# ============================================================
# SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# SCALE
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


print(
    "\nTraining risk model..."
)


model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========================================")
print(" Risk Model Evaluation")
print("========================================")

print(
    f"MAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print("\n========================================")
print(" Risk Model Saved")
print("========================================")

print(
    f"\nRisk model:\n{MODEL_PATH}"
)

print(
    f"\nScaler:\n{SCALER_PATH}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


print(
    "\nRisk Feature Importance:"
)

print(
    importance.to_string(
        index=False
    )
)


print(
    "\nRisk training completed successfully.\n"
)
