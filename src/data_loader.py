import os
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

POLLUTION_DATASET = os.path.join(
    DATA_DIR, "pollution_dataset.csv"
)

WEATHER_DATASET = os.path.join(
    DATA_DIR, "weather_dataset.csv"
)

CITY_DATASET = os.path.join(
    DATA_DIR, "city_data.csv"
)


POLLUTION_COLUMNS = [
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


def load_pollution_dataset():
    """
    Load pollution training dataset.
    """

    if not os.path.exists(POLLUTION_DATASET):
        raise FileNotFoundError(
            f"Pollution dataset not found: {POLLUTION_DATASET}"
        )

    df = pd.read_csv(POLLUTION_DATASET)

    return df


def load_weather_dataset():
    """
    Load weather dataset.
    """

    if not os.path.exists(WEATHER_DATASET):
        raise FileNotFoundError(
            f"Weather dataset not found: {WEATHER_DATASET}"
        )

    return pd.read_csv(WEATHER_DATASET)


def load_city_dataset():
    """
    Load city information.
    """

    if not os.path.exists(CITY_DATASET):
        raise FileNotFoundError(
            f"City dataset not found: {CITY_DATASET}"
        )

    return pd.read_csv(CITY_DATASET)


def clean_pollution_data(df):
    """
    Clean pollution dataframe.
    """

    df = df.copy()

    for column in POLLUTION_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            column
            for column in POLLUTION_COLUMNS
            if column in df.columns
        ]
    )

    return df


def get_city_data(city):
    """
    Return pollution/weather data for a specific city.
    """

    pollution_df = load_pollution_dataset()

    if "City" not in pollution_df.columns:
        return pollution_df

    result = pollution_df[
        pollution_df["City"].astype(str).str.lower()
        == str(city).lower()
    ]

    return result


def get_latest_pollution(city=None):
    """
    Return latest pollution observation.

    If City column exists, filter by city.
    Otherwise return the last dataset row.
    """

    df = load_pollution_dataset()

    if city and "City" in df.columns:

        city_df = df[
            df["City"].astype(str).str.lower()
            == str(city).lower()
        ]

        if not city_df.empty:
            return city_df.iloc[-1].to_dict()

    return df.iloc[-1].to_dict()


def create_feature_dataframe(data):
    """
    Convert pollution dictionary into a model-ready DataFrame.
    """

    values = {}

    for feature in POLLUTION_COLUMNS:

        value = data.get(feature, 0)

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        values[feature] = value

    return pd.DataFrame(
        [values],
        columns=POLLUTION_COLUMNS
    )
