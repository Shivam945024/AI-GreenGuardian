"""
AI-GreenGuardian
Chart utilities.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# AQI TREND
# ============================================================

def pollution_chart(df):
    """
    Create AQI trend chart.

    Expected columns:
    time
    AQI
    """

    if df is None or df.empty:
        return go.Figure()

    fig = px.line(
        df,
        x="time",
        y="AQI",
        markers=True,
        title="AQI Trend"
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="AQI",
        hovermode="x unified"
    )

    return fig


# ============================================================
# POLLUTANT BAR CHART
# ============================================================

def pollutant_bar_chart(data):
    """
    Create pollutant concentration chart.
    """

    pollutants = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3"
    ]

    values = [
        float(data.get(
            pollutant,
            0
        ))
        for pollutant in pollutants
    ]

    df = pd.DataFrame({
        "Pollutant": pollutants,
        "Value": values
    })

    fig = px.bar(
        df,
        x="Pollutant",
        y="Value",
        title="Current Pollutant Levels",
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title="Pollutant",
        yaxis_title="Concentration"
    )

    return fig


# ============================================================
# AQI GAUGE
# ============================================================

def aqi_gauge(aqi):
    """
    Create AQI gauge.
    """

    aqi = max(
        0,
        min(
            500,
            float(aqi)
        )
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi,
            title={
                "text": "AQI"
            },
            gauge={
                "axis": {
                    "range": [0, 500]
                },
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "green"
                    },
                    {
                        "range": [50, 100],
                        "color": "yellow"
                    },
                    {
                        "range": [100, 200],
                        "color": "orange"
                    },
                    {
                        "range": [200, 300],
                        "color": "red"
                    },
                    {
                        "range": [300, 500],
                        "color": "darkred"
                    }
                ],
            }
        )
    )

    fig.update_layout(
        height=350
    )

    return fig


# ============================================================
# RISK GAUGE
# ============================================================

def risk_gauge(score):
    """
    Create environmental risk gauge.
    """

    score = max(
        0,
        min(
            100,
            float(score)
        )
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={
                "text": "Environmental Risk"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "steps": [
                    {
                        "range": [0, 25],
                        "color": "green"
                    },
                    {
                        "range": [25, 50],
                        "color": "yellow"
                    },
                    {
                        "range": [50, 75],
                        "color": "orange"
                    },
                    {
                        "range": [75, 100],
                        "color": "red"
                    }
                ],
            }
        )
    )

    fig.update_layout(
        height=350
    )

    return fig


# ============================================================
# WEATHER CHART
# ============================================================

def weather_chart(df):
    """
    Create temperature/humidity chart.

    Expected columns:
    time
    Temperature
    Humidity
    """

    if df is None or df.empty:
        return go.Figure()

    fig = go.Figure()

    if "Temperature" in df.columns:

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["Temperature"],
                mode="lines+markers",
                name="Temperature"
            )
        )

    if "Humidity" in df.columns:

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["Humidity"],
                mode="lines+markers",
                name="Humidity"
            )
        )

    fig.update_layout(
        title="Weather Trend",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified"
    )

    return fig


# ============================================================
# FEATURE CONTRIBUTION
# ============================================================

def contribution_chart(
    contribution_data
):
    """
    Create explainability chart.

    Accepts:
    dictionary
    or DataFrame
    """

    if isinstance(
        contribution_data,
        dict
    ):

        df = pd.DataFrame({
            "Feature": list(
                contribution_data.keys()
            ),
            "Contribution": list(
                contribution_data.values()
            )
        })

    else:

        df = contribution_data.copy()

        if "Contribution (%)" in df.columns:

            df = df.rename(
                columns={
                    "Contribution (%)":
                        "Contribution"
                }
            )

    df = df.sort_values(
        "Contribution",
        ascending=True
    )

    fig = px.bar(
        df,
        x="Contribution",
        y="Feature",
        orientation="h",
        title="Environmental Feature Contributions",
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title="Contribution (%)",
        yaxis_title="Feature"
    )

    return fig


# ============================================================
# CITY COMPARISON
# ============================================================

def city_comparison_chart(
    df,
    city_column="City",
    value_column="AQI"
):
    """
    Compare AQI across cities.
    """

    if df is None or df.empty:
        return go.Figure()

    fig = px.bar(
        df,
        x=city_column,
        y=value_column,
        title="City AQI Comparison",
        text_auto=".1f"
    )

    fig.update_layout(
        xaxis_title="City",
        yaxis_title="AQI"
    )

    return fig


# ============================================================
# POLLUTION COMPARISON
# ============================================================

def pollutant_comparison_chart(
    df,
    city_column="City"
):
    """
    Compare pollutants across cities.
    """

    pollutants = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "O3"
    ]

    available = [
        column
        for column in pollutants
        if column in df.columns
    ]

    if not available:
        return go.Figure()

    melted = df.melt(
        id_vars=[city_column],
        value_vars=available,
        var_name="Pollutant",
        value_name="Value"
    )

    fig = px.bar(
        melted,
        x=city_column,
        y="Value",
        color="Pollutant",
        barmode="group",
        title="Pollutant Comparison"
    )

    return fig


# ============================================================
# MAP
# ============================================================

def city_map(df):
    """
    Create city pollution map.

    Expected:
    Latitude
    Longitude
    City
    AQI
    """

    required = [
        "Latitude",
        "Longitude",
        "City"
    ]

    if not all(
        column in df.columns
        for column in required
    ):

        return go.Figure()

    fig = px.scatter_map(
        df,
        lat="Latitude",
        lon="Longitude",
        hover_name="City",
        hover_data=[
            column
            for column in [
                "AQI",
                "PM2.5",
                "PM10"
            ]
            if column in df.columns
        ],
        zoom=4,
        height=500,
        title="Environmental Monitoring Map"
    )

    fig.update_layout(
        map_style="open-street-map"
    )

    return fig


# ============================================================
# HISTORY CHART
# ============================================================

def pollutant_history_chart(
    df,
    pollutants=None
):
    """
    Plot historical pollutant values.
    """

    if df is None or df.empty:
        return go.Figure()

    if pollutants is None:

        pollutants = [
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "O3"
        ]

    available = [
        pollutant
        for pollutant in pollutants
        if pollutant in df.columns
    ]

    fig = go.Figure()

    for pollutant in available:

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[pollutant],
                mode="lines",
                name=pollutant
            )
        )

    fig.update_layout(
        title="Pollutant History",
        xaxis_title="Time",
        yaxis_title="Concentration",
        hovermode="x unified"
    )

    return fig
