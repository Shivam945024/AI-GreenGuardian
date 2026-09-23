"""
AI-GreenGuardian
Alert and notification utilities.
"""


def get_aqi_alert(aqi):
    """
    Generate an alert based on AQI.
    """

    aqi = float(aqi)

    if aqi <= 50:
        return {
            "level": "LOW",
            "title": "Good Air Quality",
            "message": (
                "Air quality is currently relatively good. "
                "Normal outdoor activities can continue."
            ),
            "severity": "success",
        }

    elif aqi <= 100:
        return {
            "level": "MODERATE",
            "title": "Moderate Air Quality",
            "message": (
                "Air quality is moderate. "
                "Continue monitoring pollution levels."
            ),
            "severity": "info",
        }

    elif aqi <= 200:
        return {
            "level": "HIGH",
            "title": "Elevated Pollution",
            "message": (
                "Pollution levels are elevated. "
                "Consider reducing prolonged outdoor exposure."
            ),
            "severity": "warning",
        }

    elif aqi <= 300:
        return {
            "level": "VERY HIGH",
            "title": "Very High Pollution",
            "message": (
                "Pollution levels are very high. "
                "Limit prolonged outdoor activities when practical."
            ),
            "severity": "error",
        }

    return {
        "level": "CRITICAL",
        "title": "Severe Pollution",
        "message": (
            "Very high pollution levels detected. "
            "Follow local environmental and public-health guidance."
        ),
        "severity": "error",
    }


def get_risk_alert(risk_score):
    """
    Generate an environmental risk alert.
    """

    score = float(risk_score)

    if score < 25:
        return {
            "level": "LOW",
            "title": "Low Environmental Risk",
            "message": (
                "Environmental conditions are relatively stable."
            ),
            "severity": "success",
        }

    elif score < 50:
        return {
            "level": "MODERATE",
            "title": "Moderate Environmental Risk",
            "message": (
                "Environmental conditions should be monitored."
            ),
            "severity": "info",
        }

    elif score < 75:
        return {
            "level": "HIGH",
            "title": "High Environmental Risk",
            "message": (
                "Elevated environmental risk has been detected."
            ),
            "severity": "warning",
        }

    return {
        "level": "CRITICAL",
        "title": "Critical Environmental Risk",
        "message": (
            "Very high environmental risk detected. "
            "Monitor local environmental guidance."
        ),
        "severity": "error",
    }


def get_pollutant_alerts(data):
    """
    Check individual pollutants.
    """

    alerts = []

    pm25 = float(data.get("PM2.5", 0))
    pm10 = float(data.get("PM10", 0))
    no2 = float(data.get("NO2", 0))
    so2 = float(data.get("SO2", 0))
    co = float(data.get("CO", 0))
    o3 = float(data.get("O3", 0))

    if pm25 > 60:
        alerts.append({
            "pollutant": "PM2.5",
            "level": "HIGH",
            "message": (
                f"PM2.5 is elevated at {pm25:.1f}."
            )
        })

    if pm10 > 100:
        alerts.append({
            "pollutant": "PM10",
            "level": "HIGH",
            "message": (
                f"PM10 is elevated at {pm10:.1f}."
            )
        })

    if no2 > 40:
        alerts.append({
            "pollutant": "NO2",
            "level": "HIGH",
            "message": (
                f"NO2 is elevated at {no2:.1f}."
            )
        })

    if so2 > 40:
        alerts.append({
            "pollutant": "SO2",
            "level": "HIGH",
            "message": (
                f"SO2 is elevated at {so2:.1f}."
            )
        })

    if co > 4:
        alerts.append({
            "pollutant": "CO",
            "level": "HIGH",
            "message": (
                f"CO is elevated at {co:.2f}."
            )
        })

    if o3 > 100:
        alerts.append({
            "pollutant": "O3",
            "level": "HIGH",
            "message": (
                f"O3 is elevated at {o3:.1f}."
            )
        })

    return alerts


def get_all_alerts(aqi, risk_score, data):
    """
    Return AQI + risk + pollutant alerts.
    """

    alerts = []

    alerts.append(
        get_aqi_alert(aqi)
    )

    alerts.append(
        get_risk_alert(risk_score)
    )

    alerts.extend(
        get_pollutant_alerts(data)
    )

    return alerts


def has_critical_alerts(
    aqi,
    risk_score
):
    """
    Check whether critical conditions exist.
    """

    return (
        float(aqi) > 300
        or float(risk_score) >= 75
    )


def alert_summary(
    aqi,
    risk_score,
    data
):
    """
    Generate a simple dashboard summary.
    """

    alerts = get_all_alerts(
        aqi,
        risk_score,
        data
    )

    critical_count = sum(
        1
        for alert in alerts
        if alert.get("severity") == "error"
    )

    warning_count = sum(
        1
        for alert in alerts
        if alert.get("severity") == "warning"
    )

    return {
        "total_alerts": len(alerts),
        "critical_alerts": critical_count,
        "warning_alerts": warning_count,
        "alerts": alerts,
    }
