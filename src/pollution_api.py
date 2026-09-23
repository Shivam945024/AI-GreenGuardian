import requests
import pandas as pd


# =========================================================
# CITY COORDINATES
# =========================================================

CITY_COORDINATES = {
    "Lucknow": (26.8467, 80.9462),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kanpur": (26.4499, 80.3319),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
}


# =========================================================
# GET CITY COORDINATES
# =========================================================

def get_city_coordinates(city):

    city = str(city).strip()

    return CITY_COORDINATES.get(
        city,
        (26.8467, 80.9462)
    )


# =========================================================
# LIVE WEATHER + AIR QUALITY
# =========================================================

def build_environment_record(
    city,
    latitude=None,
    longitude=None
):

    # If coordinates are not supplied,
    # automatically get them from city.

    if latitude is None or longitude is None:

        latitude, longitude = get_city_coordinates(
            city
        )

    # -----------------------------------------------------
    # Open-Meteo API
    # -----------------------------------------------------

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "current": (
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),

        "timezone": "auto",
    }

    response = requests.get(
        url,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    current = data.get(
        "current",
        {}
    )

    # -----------------------------------------------------
    # Weather API
    # -----------------------------------------------------

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m"
        ),

        "timezone": "auto",
    }

    weather_response = requests.get(
        weather_url,
        params=weather_params,
        timeout=15,
    )

    weather_response.raise_for_status()

    weather_data = weather_response.json()

    weather_current = weather_data.get(
        "current",
        {}
    )

    # -----------------------------------------------------
    # Environment Record
    # -----------------------------------------------------

    environment = {
        "City": city,

        "PM2.5": current.get(
            "pm2_5",
            0
        ),

        "PM10": current.get(
            "pm10",
            0
        ),

        "NO2": current.get(
            "nitrogen_dioxide",
            0
        ),

        "SO2": current.get(
            "sulphur_dioxide",
            0
        ),

        "CO": current.get(
            "carbon_monoxide",
            0
        ),

        "O3": current.get(
            "ozone",
            0
        ),

        "Temperature": weather_current.get(
            "temperature_2m",
            0
        ),

        "Humidity": weather_current.get(
            "relative_humidity_2m",
            0
        ),

        "Wind Speed": weather_current.get(
            "wind_speed_10m",
            0
        ),
    }

    return environment


# =========================================================
# HISTORICAL AIR QUALITY
# =========================================================

def air_quality_history(
    city,
    days=7
):

    latitude, longitude = get_city_coordinates(
        city
    )

    url = (
        "https://air-quality-api.open-meteo.com/"
        "v1/air-quality"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": (
            "pm10,"
            "pm2_5,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "carbon_monoxide,"
            "ozone"
        ),

        "past_days": days,

        "timezone": "auto",
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    hourly = data.get(
        "hourly",
        {}
    )

    if not hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"]
        )

    return df
