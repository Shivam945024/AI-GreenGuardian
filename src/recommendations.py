def recommendations(
    aqi,
    risk,
    data
):
    """
    Generate environmental recommendations.
    """

    recommendations_list = []

    aqi = float(aqi)

    pm25 = float(
        data.get("PM2.5", 0)
    )

    pm10 = float(
        data.get("PM10", 0)
    )

    no2 = float(
        data.get("NO2", 0)
    )

    wind = float(
        data.get("Wind Speed", 0)
    )

    temperature = float(
        data.get("Temperature", 0)
    )

    humidity = float(
        data.get("Humidity", 0)
    )

    # AQI recommendation
    if aqi <= 50:

        recommendations_list.append(
            "Air quality is relatively good. "
            "Continue normal outdoor activities."
        )

    elif aqi <= 100:

        recommendations_list.append(
            "Air quality is moderate. "
            "Continue monitoring local air-quality conditions."
        )

    elif aqi <= 200:

        recommendations_list.append(
            "Pollution is elevated. "
            "Consider reducing prolonged outdoor exposure."
        )

    elif aqi <= 300:

        recommendations_list.append(
            "Pollution is very high. "
            "Limit prolonged outdoor activities when practical."
        )

    else:

        recommendations_list.append(
            "Pollution is severe. "
            "Follow local public-health and environmental guidance."
        )

    # PM2.5
    if pm25 > 60:

        recommendations_list.append(
            "PM2.5 is elevated. "
            "Monitor particulate pollution closely."
        )

    # PM10
    if pm10 > 100:

        recommendations_list.append(
            "PM10 is elevated. "
            "Dust and particulate exposure should be minimized."
        )

    # NO2
    if no2 > 40:

        recommendations_list.append(
            "NO2 levels are elevated. "
            "Vehicle and combustion-related pollution may be contributing."
        )

    # Wind
    if wind < 5:

        recommendations_list.append(
            "Low wind conditions may reduce pollutant dispersion."
        )

    # Temperature
    if temperature > 35:

        recommendations_list.append(
            "High temperature detected. "
            "Stay hydrated and monitor heat conditions."
        )

    # Humidity
    if humidity > 85:

        recommendations_list.append(
            "High humidity detected. "
            "Monitor changing environmental conditions."
        )

    # Risk
    if risk in [
        "HIGH",
        "CRITICAL"
    ]:

        recommendations_list.append(
            "Environmental risk is elevated. "
            "Monitor the dashboard and local environmental alerts."
        )

    # General sustainability
    recommendations_list.append(
        "Use public transport, walking, cycling, "
        "or shared mobility when practical to reduce emissions."
    )

    return recommendations_list


def short_recommendation(
    aqi,
    risk
):
    """
    Generate a short dashboard message.
    """

    if risk == "CRITICAL":
        return (
            "Critical environmental risk detected. "
            "Monitor local guidance."
        )

    if risk == "HIGH":
        return (
            "High environmental risk detected. "
            "Reduce unnecessary pollution exposure."
        )

    if aqi > 100:
        return (
            "Air pollution is elevated. "
            "Monitor conditions regularly."
        )

    return (
        "Environmental conditions are currently "
        "relatively stable."
    )
