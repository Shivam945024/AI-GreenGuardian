"""
AI-GreenGuardian
General helper functions.
"""

import os
import json
from datetime import datetime

import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


def project_path(*paths):
    """
    Build an absolute path inside project.
    """

    return os.path.join(
        BASE_DIR,
        *paths
    )


# ============================================================
# NUMBER HELPERS
# ============================================================

def safe_float(
    value,
    default=0.0
):
    """
    Safely convert value to float.
    """

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return default


def safe_int(
    value,
    default=0
):
    """
    Safely convert value to integer.
    """

    try:

        if value is None:
            return default

        return int(float(value))

    except (
        TypeError,
        ValueError
    ):

        return default


def clamp(
    value,
    minimum,
    maximum
):
    """
    Keep a value within a range.
    """

    value = safe_float(
        value
    )

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def format_number(
    value,
    decimals=2
):
    """
    Format numeric value.
    """

    value = safe_float(
        value
    )

    return f"{value:,.{decimals}f}"


# ============================================================
# AQI HELPERS
# ============================================================

def get_aqi_status(aqi):
    """
    Return simple AQI status.
    """

    aqi = safe_float(aqi)

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 200:
        return "Poor"

    if aqi <= 300:
        return "Very Poor"

    return "Severe"


def get_aqi_emoji(aqi):
    """
    Return an AQI UI emoji.
    """

    category = get_aqi_status(
        aqi
    )

    emojis = {
        "Good": "🟢",
        "Moderate": "🟡",
        "Poor": "🟠",
        "Very Poor": "🔴",
        "Severe": "🟣"
    }

    return emojis.get(
        category,
        "⚪"
    )


def get_risk_status(score):
    """
    Return environmental risk status.
    """

    score = safe_float(score)

    if score < 25:
        return "LOW"

    if score < 50:
        return "MODERATE"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


# ============================================================
# DATA HELPERS
# ============================================================

def clean_dataframe(df):
    """
    Clean common dataframe problems.
    """

    if df is None:
        return pd.DataFrame()

    df = df.copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


def numeric_columns(
    df
):
    """
    Return numeric columns.
    """

    if df is None or df.empty:
        return []

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def dataframe_to_csv(
    df
):
    """
    Convert dataframe to CSV bytes.
    """

    if df is None:
        return b""

    return df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


# ============================================================
# DICTIONARY HELPERS
# ============================================================

def clean_environment_data(data):
    """
    Normalize environment dictionary.
    """

    fields = [
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

    cleaned = {}

    for field in fields:

        cleaned[field] = safe_float(
            data.get(field, 0)
        )

    return cleaned


def dictionary_to_dataframe(
    data
):
    """
    Convert dictionary into one-row DataFrame.
    """

    return pd.DataFrame(
        [data]
    )


# ============================================================
# DATE/TIME HELPERS
# ============================================================

def current_timestamp():
    """
    Return current timestamp.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def format_datetime(
    value
):
    """
    Format datetime value.
    """

    try:

        dt = pd.to_datetime(
            value
        )

        return dt.strftime(
            "%d %b %Y, %I:%M %p"
        )

    except Exception:

        return str(value)


# ============================================================
# JSON HELPERS
# ============================================================

def save_json(
    data,
    filepath
):
    """
    Save dictionary as JSON.
    """

    directory = os.path.dirname(
        filepath
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


def load_json(
    filepath,
    default=None
):
    """
    Load JSON safely.
    """

    if not os.path.exists(
        filepath
    ):

        return (
            {}
            if default is None
            else default
        )

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return (
            {}
            if default is None
            else default
        )


# ============================================================
# FILE HELPERS
# ============================================================

def ensure_directory(
    directory
):
    """
    Create directory if necessary.
    """

    os.makedirs(
        directory,
        exist_ok=True
    )

    return directory


def file_exists(
    filepath
):
    """
    Check whether a file exists.
    """

    return os.path.isfile(
        filepath
    )


# ============================================================
# MODEL STATUS
# ============================================================

def model_status(
    model_path
):
    """
    Return model availability status.
    """

    if os.path.exists(
        model_path
    ):

        return {
            "available": True,
            "status": "READY",
            "path": model_path
        }

    return {
        "available": False,
        "status": "NOT FOUND",
        "path": model_path
    }


# ============================================================
# CITY HELPERS
# ============================================================

def find_city(
    df,
    city
):
    """
    Find city from city dataframe.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if "City" not in df.columns:
        return pd.DataFrame()

    return df[
        df["City"]
        .astype(str)
        .str.lower()
        .str.strip()
        ==
        str(city)
        .lower()
        .strip()
    ]


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

def dashboard_summary(
    city,
    aqi,
    predicted_aqi,
    risk_score
):
    """
    Create a dashboard summary dictionary.
    """

    change = (
        safe_float(predicted_aqi)
        - safe_float(aqi)
    )

    return {
        "City": city,
        "Current AQI": round(
            safe_float(aqi),
            2
        ),
        "Predicted AQI": round(
            safe_float(predicted_aqi),
            2
        ),
        "AQI Change": round(
            change,
            2
        ),
        "Risk Score": round(
            safe_float(risk_score),
            2
        ),
        "AQI Status": get_aqi_status(
            aqi
        ),
        "Risk Status": get_risk_status(
            risk_score
        ),
        "Updated": current_timestamp()
    }
