import math


def _safe(value, default=0.0):

    try:

        if value is None:
            return default

        value = float(value)

        if math.isnan(value):
            return default

        return value

    except (TypeError, ValueError):

        return default


def environmental_risk(environment):

    pm25 = _safe(environment.get("PM2.5"))
    pm10 = _safe(environment.get("PM10"))
    no2 = _safe(environment.get("NO2"))
    so2 = _safe(environment.get("SO2"))
    co = _safe(environment.get("CO"))
    o3 = _safe(environment.get("O3"))

    # Normalize pollutant contribution
    pm25_score = min((pm25 / 60) * 100, 100)
    pm10_score = min((pm10 / 100) * 100, 100)
    no2_score = min((no2 / 80) * 100, 100)
    so2_score = min((so2 / 80) * 100, 100)
    co_score = min((co / 10000) * 100, 100)
    o3_score = min((o3 / 120) * 100, 100)

    # Weighted risk
    risk = (

        pm25_score * 0.35
        + pm10_score * 0.20
        + no2_score * 0.15
        + so2_score * 0.10
        + co_score * 0.10
        + o3_score * 0.10

    )

    return round(
        max(0, min(risk, 100)),
        2
    )


def risk_level(score):

    score = _safe(score)

    if score <= 20:
        return "Low"

    elif score <= 40:
        return "Moderate"

    elif score <= 60:
        return "High"

    elif score <= 80:
        return "Very High"

    else:
        return "Critical"


def risk_message(score):

    level = risk_level(score)

    messages = {

        "Low":
            "Environmental conditions are generally safe.",

        "Moderate":
            "Some pollution is present. Sensitive people should take care.",

        "High":
            "Pollution levels may affect health. Reduce prolonged outdoor exposure.",

        "Very High":
            "High environmental risk. Limit outdoor activities.",

        "Critical":
            "Critical pollution conditions. Avoid unnecessary outdoor exposure."
    }

    return messages[level]
