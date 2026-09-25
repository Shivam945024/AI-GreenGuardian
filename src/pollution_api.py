import requests
import pandas as pd
import time


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


def get_city_coordinates(city):

    city = str(city).strip().lower()

    for name, coords in CITY_COORDINATES.items():

        if name.lower() == city:
            return coords

    return CITY_COORDINATES["Lucknow"]


def safe_request(url, params, retries=1):

    for attempt in range(retries + 1):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=15
            )

            if response.status_code == 429:

                if attempt < retries:
                    time.sleep(3)
                    continue

                return None

            response.raise_for_status()

            return response.json()

        except requests.RequestException:

            if attempt < retries:
                time.sleep(2)
                continue

            return None

    return None


def build_environment_record(
    city,
    latitude=None,
    longitude=None
):

    if latitude is None or longitude is None:
        latitude, longitude = get_city_coordinates(city)

    # ---------------------------------------------
    # AIR QUALITY
    # ---------------------------------------------

    air_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    air_params = {

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

        "timezone": "auto"
    }

    air_data = safe_request(
        air_url,
        air_params
    )

    # ---------------------------------------------
    # WEATHER
    # ---------------------------------------------

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

        "timezone": "auto"
    }

    weather_data = safe_request(
        weather_url,
        weather_params
    )

    # ---------------------------------------------
    # ENVIRONMENT DATA
    # ---------------------------------------------

    environment = {
        "City": city,

        "PM2.5": None,
        "PM10": None,
        "NO2": None,
        "SO2": None,
        "CO": None,
        "O3": None,

        "Temperature": None,
        "Humidity": None,
        "Wind Speed": None,
    }

    # ---------------------------------------------
    # AIR DATA
    # ---------------------------------------------

    if air_data:

        current = air_data.get(
            "current",
            {}
        )

        environment["PM2.5"] = current.get("pm2_5")
        environment["PM10"] = current.get("pm10")
        environment["NO2"] = current.get("nitrogen_dioxide")
        environment["SO2"] = current.get("sulphur_dioxide")
        environment["CO"] = current.get("carbon_monoxide")
        environment["O3"] = current.get("ozone")

    # ---------------------------------------------
    # WEATHER DATA
    # ---------------------------------------------

    if weather_data:

        current_weather = weather_data.get(
            "current",
            {}
        )

        environment["Temperature"] = current_weather.get(
            "temperature_2m"
        )

        environment["Humidity"] = current_weather.get(
            "relative_humidity_2m"
        )

        environment["Wind Speed"] = current_weather.get(
            "wind_speed_10m"
        )

    return environment


def air_quality_history(city, days=7):

    latitude, longitude = get_city_coordinates(city)

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
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

        "timezone": "auto"
    }

    data = safe_request(
        url,
        params
    )

    if not data:
        return pd.DataFrame()

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
