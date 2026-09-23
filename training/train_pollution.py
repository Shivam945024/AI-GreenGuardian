import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    "pollution_model.pkl"
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

TARGET = "AQI"


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========================================")
print(" AI-GreenGuardian Pollution Model")
print("========================================\n")

print(
    f"Loading dataset:\n{DATASET_PATH}"
)

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n"
    )


df = pd.read_csv(
    DATASET_PATH
)


print(
    f"\nDataset shape: {df.shape}"
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df[
    required_columns
].copy()


for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna()


# Remove impossible negative values
for column in FEATURES:

    df = df[
        df[column] >= 0
    ]


df = df[
    (df[TARGET] >= 0)
    &
    (df[TARGET] <= 500)
]


print(
    f"\nClean dataset shape: {df.shape}"
)


# ============================================================
# CHECK DATA SIZE
# ============================================================

if len(df) < 30:

    raise ValueError(
        "\nDataset contains too few rows.\n"
        "Please provide at least 30 observations "
        "for training.\n"
    )


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[
    FEATURES
]

y = df[
    TARGET
]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# RANDOM FOREST MODEL
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
    "\nTraining Random Forest..."
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
print(" Pollution Model Evaluation")
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
print(" Model Saved Successfully")
print("========================================")

print(
    f"\nPollution model:\n{MODEL_PATH}"
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
    "\nFeature Importance:"
)

print(
    importance.to_string(
        index=False
    )
)


print(
    "\nTraining completed successfully.\n"
)
