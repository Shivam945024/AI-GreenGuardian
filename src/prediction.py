import os
import joblib
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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


FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity",
    "Wind Speed",
]


def load_models():
    """
    Load all trained ML models.
    """

    pollution_model = None
    risk_model = None
    scaler = None

    if os.path.exists(
        POLLUTION_MODEL_PATH
    ):
        pollution_model = joblib.load(
            POLLUTION_MODEL_PATH
        )

    if os.path.exists(
        RISK_MODEL_PATH
    ):
        risk_model = joblib.load(
            RISK_MODEL_PATH
        )

    if os.path.exists(
        SCALER_PATH
    ):
        scaler = joblib.load(
            SCALER_PATH
        )

    return (
        pollution_model,
        risk_model,
        scaler
    )


def prepare_features(data):
    """
    Convert input dictionary into model features.
    """

    values = []

    for feature in FEATURES:

        value = data.get(
            feature,
            0
        )

        try:
            value = float(value)
        except (
            TypeError,
            ValueError
        ):
            value = 0.0

        values.append(value)

    return pd.DataFrame(
        [values],
        columns=FEATURES
    )


def predict_pollution(data):
    """
    Predict AQI/pollution using trained model.
    """

    model, _, scaler = load_models()

    X = prepare_features(data)

    if model is None:
        return None

    if scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    prediction = model.predict(
        X_input
    )

    return round(
        float(prediction[0]),
        2
    )


def predict_risk(data):
    """
    Predict environmental risk using risk model.
    """

    _, model, scaler = load_models()

    if model is None:
        return None

    X = prepare_features(data)

    if scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    prediction = model.predict(
        X_input
    )

    value = float(
        prediction[0]
    )

    return round(
        max(0, min(100, value)),
        2
    )


def predict_next_aqi(data, current_aqi=None):
    """
    Fallback AQI prediction when a trained model
    is unavailable.
    """

    model_prediction = predict_pollution(
        data
    )

    if model_prediction is not None:
        return model_prediction

    if current_aqi is None:
        current_aqi = 0

    pm25 = float(
        data.get("PM2.5", 0)
    )

    wind = float(
        data.get("Wind Speed", 0)
    )

    adjustment = (
        pm25 * 0.15
        - wind * 0.5
    )

    prediction = (
        float(current_aqi)
        + adjustment
    )

    return round(
        max(0, min(500, prediction)),
        2
    )


def prediction_change(
    current_aqi,
    predicted_aqi
):
    """
    Calculate AQI prediction change.
    """

    current_aqi = float(current_aqi)
    predicted_aqi = float(predicted_aqi)

    change = (
        predicted_aqi
        - current_aqi
    )

    percentage = (
        (change / current_aqi) * 100
        if current_aqi != 0
        else 0
    )

    return {
        "absolute": round(
            change,
            2
        ),
        "percentage": round(
            percentage,
            2
        ),
        "direction": (
            "increase"
            if change > 0
            else "decrease"
            if change < 0
            else "stable"
        ),
    }
