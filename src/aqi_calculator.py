def calculate_aqi(
    pm25,
    pm10,
    no2,
    so2,
    co,
    o3
):
    """
    Prototype AQI-style environmental index.

    IMPORTANT:
    This is not an official CPCB AQI formula.
    """

    pm25 = max(float(pm25), 0)
    pm10 = max(float(pm10), 0)
    no2 = max(float(no2), 0)
    so2 = max(float(so2), 0)
    co = max(float(co), 0)
    o3 = max(float(o3), 0)

    score = (
        pm25 * 1.8
        + pm10 * 0.7
        + no2 * 0.5
        + so2 * 0.3
        + co * 8
        + o3 * 0.4
    )

    aqi = score / 3.7

    return round(
        max(0, min(500, aqi)),
        2
    )


def aqi_category(aqi):
    """
    Convert AQI score into category.
    """

    aqi = float(aqi)

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 200:
        return "Poor"

    if aqi <= 300:
        return "Very Poor"

    return "Severe"


def aqi_description(category):
    """
    Human-readable AQI description.
    """

    descriptions = {
        "Good":
            "Air quality is relatively clean.",

        "Moderate":
            "Air quality is acceptable but pollution should be monitored.",

        "Poor":
            "Pollution is elevated and sensitive people should take precautions.",

        "Very Poor":
            "High pollution levels require increased precaution.",

        "Severe":
            "Very high pollution levels require strong precautionary measures.",
    }

    return descriptions.get(
        category,
        "Monitor local air-quality conditions."
    )


def aqi_color(category):
    """
    UI-friendly category color.
    """

    colors = {
        "Good": "green",
        "Moderate": "yellow",
        "Poor": "orange",
        "Very Poor": "red",
        "Severe": "darkred",
    }

    return colors.get(
        category,
        "gray"
    )
