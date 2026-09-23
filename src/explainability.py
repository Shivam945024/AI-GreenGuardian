import pandas as pd


FEATURES = [
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


def feature_contributions(data):
    """
    Estimate interpretable feature contributions.

    This is a heuristic explanation and not a true SHAP explanation.
    """

    values = {
        "PM2.5": max(
            0,
            float(data.get("PM2.5", 0))
        ),

        "PM10": max(
            0,
            float(data.get("PM10", 0))
        ),

        "NO2": max(
            0,
            float(data.get("NO2", 0))
        ),

        "SO2": max(
            0,
            float(data.get("SO2", 0))
        ),

        "CO": max(
            0,
            float(data.get("CO", 0))
        ),

        "O3": max(
            0,
            float(data.get("O3", 0))
        ),

        "Temperature": max(
            0,
            float(data.get("Temperature", 0))
        ),

        "Humidity": max(
            0,
            float(data.get("Humidity", 0))
        ),

        "Wind Speed": max(
            0,
            20 - float(
                data.get("Wind Speed", 0)
            )
        ),
    }

    total = sum(
        values.values()
    )

    if total == 0:
        total = 1

    contributions = {
        key: round(
            (value / total) * 100,
            2
        )
        for key, value in values.items()
    }

    return contributions


def contribution_dataframe(data):
    """
    Return contributions as a DataFrame.
    """

    contributions = feature_contributions(
        data
    )

    df = pd.DataFrame({
        "Feature": list(
            contributions.keys()
        ),
        "Contribution (%)": list(
            contributions.values()
        ),
    })

    return df.sort_values(
        "Contribution (%)",
        ascending=False
    )


def top_contributors(
    data,
    n=3
):
    """
    Return the most influential heuristic factors.
    """

    contributions = feature_contributions(
        data
    )

    sorted_features = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_features[:n]


def explain_prediction(
    data,
    predicted_aqi
):
    """
    Generate a human-readable explanation.
    """

    top_features = top_contributors(
        data,
        n=3
    )

    explanation = [
        f"Predicted AQI: {predicted_aqi}"
    ]

    if top_features:

        names = ", ".join(
            feature
            for feature, _ in top_features
        )

        explanation.append(
            f"Key contributing environmental "
            f"factors: {names}."
        )

    explanation.append(
        "These contributions are heuristic "
        "indicators and should not be interpreted "
        "as causal effects."
    )

    return " ".join(
        explanation
    )
