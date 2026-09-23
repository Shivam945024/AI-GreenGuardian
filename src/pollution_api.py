import requests
import pandas as pd


AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


def get_air_quality(
    latitude,
    longitude,
    forecast_days=1
):
    """
    Get current/hourly air-quality information.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


def get_weather(
    latitude,
    longitude,
    forecast_days=1
):
    """
    Get weather information.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "surface_pressure,"
            "precipitation"
        ),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


def get_combined_environment_data(
    latitude,
    longitude
):
    """
    Combine air-quality and weather API data.
    """

    air_data = get_air_quality(
        latitude,
        longitude
    )

    weather_data = get_weather(
        latitude,
        longitude
    )

    return {
        "air_quality": air_data,
        "weather": weather_data
    }


def extract_latest_air_quality(data):
    """
    Extract the latest available pollution values.
    """

    hourly = data.get("hourly", {})

    def latest(key):
        values = hourly.get(key, [])

        if not values:
            return 0.0

        value = values[-1]

        return float(value) if value is not None else 0.0

    return {
        "PM2.5": latest("pm2_5"),
        "PM10": latest("pm10"),
        "CO": latest("carbon_monoxide"),
        "NO2": latest("nitrogen_dioxide"),
        "SO2": latest("sulphur_dioxide"),
        "O3": latest("ozone"),
    }


def extract_latest_weather(data):
    """
    Extract latest weather values.
    """

    hourly = data.get("hourly", {})

    def latest(key):
        values = hourly.get(key, [])

        if not values:
            return 0.0

        value = values[-1]

        return float(value) if value is not None else 0.0

    return {
        "Temperature": latest("temperature_2m"),
        "Humidity": latest(
            "relative_humidity_2m"
        ),
        "Wind Speed": latest(
            "wind_speed_10m"
        ),
        "Pressure": latest(
            "surface_pressure"
        ),
        "Rainfall": latest(
            "precipitation"
        ),
    }


def build_environment_record(
    latitude,
    longitude
):
    """
    Return a single model-ready environment record.
    """

    combined = get_combined_environment_data(
        latitude,
        longitude
    )

    pollution = extract_latest_air_quality(
        combined["air_quality"]
    )

    weather = extract_latest_weather(
        combined["weather"]
    )

    return {
        **pollution,
        **weather
    }


def air_quality_history(
    latitude,
    longitude
):
    """
    Return hourly pollution history as DataFrame.
    """

    data = get_air_quality(
        latitude,
        longitude,
        forecast_days=2
    )

    hourly = data.get("hourly", {})

    df = pd.DataFrame({
        "time": hourly.get("time", []),
        "PM2.5": hourly.get("pm2_5", []),
        "PM10": hourly.get("pm10", []),
        "NO2": hourly.get(
            "nitrogen_dioxide", []
        ),
        "SO2": hourly.get(
            "sulphur_dioxide", []
        ),
        "CO": hourly.get(
            "carbon_monoxide", []
        ),
        "O3": hourly.get(
            "ozone", []
        ),
    })

    return df
