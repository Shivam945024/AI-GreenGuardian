def calculate_risk_score(
    aqi,
    pm25,
    pm10,
    no2,
    temperature,
    humidity,
    wind_speed
):
    """
    Calculate environmental risk score from 0-100.
    """

    aqi_component = min(
        100,
        float(aqi) / 5
    )

    pm25_component = min(
        100,
        float(pm25) / 2
    )

    pm10_component = min(
        100,
        float(pm10) / 3
    )

    no2_component = min(
        100,
        float(no2) / 2
    )

    temperature_component = min(
        100,
        max(
            0,
            (float(temperature) - 20) * 3
        )
    )

    humidity_component = (
        100
        if humidity > 85
        else max(0, humidity - 60) * 2
    )

    wind_component = max(
        0,
        100 - float(wind_speed) * 10
    )

    score = (
        aqi_component * 0.35
        + pm25_component * 0.20
        + pm10_component * 0.10
        + no2_component * 0.10
        + temperature_component * 0.05
        + humidity_component * 0.05
        + wind_component * 0.15
    )

    return round(
        max(0, min(100, score)),
        2
    )


def risk_level(score):
    """
    Convert risk score to risk category.
    """

    score = float(score)

    if score < 25:
        return "LOW"

    if score < 50:
        return "MODERATE"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


def environmental_risk(data, aqi):
    """
    Complete risk calculation.
    """

    score = calculate_risk_score(
        aqi=aqi,
        pm25=data.get("PM2.5", 0),
        pm10=data.get("PM10", 0),
        no2=data.get("NO2", 0),
        temperature=data.get(
            "Temperature",
            0
        ),
        humidity=data.get(
            "Humidity",
            0
        ),
        wind_speed=data.get(
            "Wind Speed",
            0
        ),
    )

    level = risk_level(score)

    return level, score


def risk_message(level):
    """
    Generate risk explanation.
    """

    messages = {
        "LOW":
            "Environmental conditions are currently relatively stable.",

        "MODERATE":
            "Environmental conditions require regular monitoring.",

        "HIGH":
            "Elevated environmental risk detected. Monitor pollution trends closely.",

        "CRITICAL":
            "Very high environmental risk detected. Follow local environmental guidance.",
    }

    return messages.get(
        level,
        "Monitor environmental conditions."
    )
