import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_csv(filename):
    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


def load_pollution_data():
    return _load_csv("pollution_dataset.csv")


def load_weather_data():
    return _load_csv("weather_dataset.csv")


def load_city_data():
    return _load_csv("city_data.csv")


def get_city_data(city_df, city_name):

    if city_df is None or city_df.empty:
        return pd.Series(dtype=object)

    possible_columns = [
        "City",
        "city",
        "CITY",
        "Name",
        "name"
    ]

    city_column = None

    for column in possible_columns:
        if column in city_df.columns:
            city_column = column
            break

    if city_column is None:
        return pd.Series(dtype=object)

    result = city_df[
        city_df[city_column]
        .astype(str)
        .str.strip()
        .str.lower()
        == str(city_name).strip().lower()
    ]

    if result.empty:
        return pd.Series(dtype=object)

    return result.iloc[0]
