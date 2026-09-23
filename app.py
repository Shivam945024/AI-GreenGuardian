```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI GreenGuardian",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero {
        padding: 30px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #063b27 0%,
                #087f5b 50%,
                #20a06a 100%
            );
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 17px;
        opacity: 0.9;
    }

    .glass-card {
        padding: 20px;
        border-radius: 20px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.05);
        margin-bottom: 15px;
    }

    .risk-high {
        padding: 18px;
        border-radius: 16px;
        background: rgba(220,53,69,0.12);
        border: 1px solid rgba(220,53,69,0.35);
    }

    .risk-medium {
        padding: 18px;
        border-radius: 16px;
        background: rgba(255,193,7,0.12);
        border: 1px solid rgba(255,193,7,0.35);
    }

    .risk-low {
        padding: 18px;
        border-radius: 16px;
        background: rgba(25,135,84,0.12);
        border: 1px solid rgba(25,135,84,0.35);
    }

    .small-text {
        font-size: 13px;
        opacity: 0.7;
    }

    div[data-testid="stMetric"] {
        border-radius: 16px;
        padding: 15px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.04);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CITY DATABASE
# ============================================================

CITIES = {
    "Lucknow": (26.8467, 80.9462),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kanpur": (26.4499, 80.3319),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873)
}


# ============================================================
# MODEL LOADER
# ============================================================

MODEL_PATH = Path("models/pollution_model.pkl")
FEATURE_PATH = Path("models/features.pkl")


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None, None

    model = joblib.load(MODEL_PATH)

    if FEATURE_PATH.exists():
        features = joblib.load(FEATURE_PATH)
    else:
        features = [
            "PM.5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "O3",
            "Temperature",
            "Humidity",
            "Wind Speed"
        ]

    return model, features


model, model_features = load_model()


# ============================================================
# DEMO DATA
# ============================================================

def get_demo_data(city):

    seed = sum(ord(c) for c in city)

    rng = np.random.default_rng(seed)

    base_aqi = {
        "Lucknow": 145,
        "Delhi": 220,
        "Mumbai": 90,
        "Bengaluru": 68,
        "Kanpur": 155,
        "Kolkata": 135,
        "Chennai": 78,
        "Hyderabad": 82,
        "Pune": 75,
        "Jaipur": 130
    }.get(city, 120)

    hours = pd.date_range(
        end=pd.Timestamp.now(),
        periods=24,
        freq="h"
    )

    trend = (
        base_aqi
        + rng.normal(0, 18, 24)
        + np.sin(np.arange(24) / 4) * 10
    )

    trend = np.clip(trend, 20, 450)

    return {
        "PM2.5": max(5, base_aqi * 0.45 + rng.normal(0, 7)),
        "PM10": max(10, base_aqi * 0.85 + rng.normal(0, 12)),
        "NO2": max(3, base_aqi * 0.18 + rng.normal(0, 4)),
        "SO2": max(1, base_aqi * 0.06 + rng.normal(0, 2)),
        "CO": max(0.1, base_aqi * 0.004 + rng.normal(0, 0.15)),
        "O3": max(5, base_aqi * 0.25 + rng.normal(0, 6)),
        "Temperature": 28 + rng.normal(0, 3),
        "Humidity": np.clip(
            58 + rng.normal(0, 12),
            20,
            95
        ),
        "Wind Speed": np.clip(
            8 + rng.normal(0, 3),
            0.5,
            25
        ),
        "history": pd.DataFrame({
            "time": hours,
            "AQI": trend
        })
    }


# ============================================================
# LIVE AIR QUALITY
# ============================================================

def get_live_air_quality(lat, lon):

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&hourly=pm2_5,pm10,carbon_monoxide,"
        "nitrogen_dioxide,sulphur_dioxide,ozone"
        "&forecast_days=1"
        "&timezone=auto"
    )

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    hourly = response.json()["hourly"]

    i = -1

    history = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "AQI": np.array(hourly["pm2_5"]) * 2
    })

    return {
        "PM2.5": float(hourly["pm2_5"][i]),
        "PM10": float(hourly["pm10"][i]),
        "NO2": float(hourly["nitrogen_dioxide"][i]),
        "SO2": float(hourly["sulphur_dioxide"][i]),
        "CO": float(hourly["carbon_monoxide"][i]) / 1000,
        "O3": float(hourly["ozone"][i]),
        "Temperature": 28.0,
        "Humidity": 60.0,
        "Wind Speed": 8.0,
        "history": history
    }


# ============================================================
# AQI CALCULATOR
# ============================================================

def calculate_aqi(pm25, pm10, no2, o3):

    value = (
        pm25 * 2.0
        + pm10
        + no2
        + o3 * 0.8
    ) / 4

    return max(
        0,
        min(500, value)
    )


def aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 200:
        return "Poor"

    if aqi <= 300:
        return "Very Poor"

    return "Severe"


# ============================================================
# ENVIRONMENTAL RISK ENGINE
# ============================================================

def environmental_risk(data, aqi):

    score = (
        0.45 * (aqi / 5)
        + 0.30 * (data["PM2.5"] / 2)
        + 0.15 * (data["NO2"] / 2)
        + 0.10 * max(
            0,
            20 - data["Wind Speed"]
        )
    )

    score = min(
        100,
        score
    )

    if score < 25:
        label = "LOW"

    elif score < 50:
        label = "MODERATE"

    elif score < 75:
        label = "HIGH"

    else:
        label = "CRITICAL"

    return label, score


# ============================================================
# ML PREDICTION
# ============================================================

def predict_aqi(data):

    if model is not None and model_features:

        try:

            values = {}

            for feature in model_features:

                if feature in data:
                    values[feature] = data[feature]

                elif feature == "PM2.5":
                    values[feature] = data["PM2.5"]

                else:
                    values[feature] = 0

            X = pd.DataFrame([values])

            prediction = model.predict(X)[0]

            return max(
                0,
                min(500, float(prediction))
            )

        except Exception:
            pass

    history = data["history"]["AQI"].tail(6).tolist()

    if len(history) >= 2:

        trend = (
            history[-1] - history[0]
        ) / (len(history) - 1)

    else:
        trend = 0

    prediction = (
        history[-1]
        + trend * 2
        + data["PM2.5"] * 0.15
    )

    return max(
        0,
        min(500, prediction)
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

def feature_contributions(data):

    values = {
        "PM2.5": data["PM2.5"],
        "PM10": data["PM10"],
        "NO₂": data["NO2"],
        "SO₂": data["SO2"],
        "O₃": data["O3"],
        "Low Wind": max(
            0,
            20 - data["Wind Speed"]
        )
    }

    total = sum(values.values())

    if total == 0:
        return values

    return {
        key: value / total * 100
        for key, value in values.items()
    }


# ============================================================
# RECOMMENDATIONS
# ============================================================

def get_recommendations(aqi, data):

    recommendations = []

    if aqi > 200:

        recommendations.append(
            "🚨 Pollution is very high. "
            "Follow current local environmental and public-health guidance."
        )

    elif aqi > 100:

        recommendations.append(
            "⚠️ Air quality is degraded. "
            "Consider limiting prolonged strenuous outdoor activity."
        )

    else:

        recommendations.append(
            "✅ Air quality is comparatively lower on this prototype scale."
        )

    if data["PM2.5"] > 60:

        recommendations.append(
            "🌫 PM2.5 is elevated. Monitor particulate pollution closely."
        )

    if data["Wind Speed"] < 5:

        recommendations.append(
            "💨 Low wind conditions may reduce pollutant dispersion."
        )

    recommendations.append(
        "🚲 Prefer public transport, walking or cycling when practical."
    )

    recommendations.append(
        "🌱 Reduce unnecessary vehicle idling and energy consumption."
    )

    return recommendations


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌱 GreenGuardian")

    st.caption(
        "AI Environmental Intelligence Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🌫 Pollution Monitor",
            "🤖 AI Prediction",
            "⚠️ Risk Intelligence",
            "🏙 City Comparison",
            "🧠 Explainable AI",
            "💡 Recommendations",
            "📄 Data Export"
        ]
    )

    st.divider()

    city = st.selectbox(
        "📍 Monitoring City",
        list(CITIES.keys())
    )

    live_mode = st.toggle(
        "🌐 Live Air Quality",
        value=False
    )

    st.divider()

    if model is not None:

        st.success(
            "🤖 ML Model Loaded"
        )

    else:

        st.warning(
            "⚠️ ML model not found"
        )

    st.caption(
        "AI GreenGuardian v1.0"
    )


# ============================================================
# LOAD DATA
# ============================================================

lat, lon = CITIES[city]

if live_mode:

    try:

        data = get_live_air_quality(
            lat,
            lon
        )

    except Exception:

        st.warning(
            "Live API unavailable. "
            "Using demo data."
        )

        data = get_demo_data(city)

else:

    data = get_demo_data(city)


# ============================================================
# CALCULATIONS
# ============================================================

aqi = calculate_aqi(
    data["PM2.5"],
    data["PM10"],
    data["NO2"],
    data["O3"]
)

category = aqi_category(aqi)

risk, risk_score = environmental_risk(
    data,
    aqi
)

predicted_aqi = predict_aqi(
    data
)

contributions = feature_contributions(
    data
)

recommendations = get_recommendations(
    aqi,
    data
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        f"""
        <div class="hero">

        <h1>🌱 AI GreenGuardian</h1>

        <p>
        Smart Pollution & Environmental Risk System
        · Monitoring: <b>{city}</b>
        </p>

        </div>
        """,
        unsafe
```
