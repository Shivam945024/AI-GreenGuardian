"""
AI-GreenGuardian
Chart utilities
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# HELPER
# ============================================================

def _safe_number(value, default=0.0):

    try:

        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

            if value == "":
                return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def _empty_chart(title="No data available"):

    fig = go.Figure()

    fig.add_annotation(
        text=title,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=18)
    )

    fig.update_layout(
        height=400,
        template="plotly_dark"
    )

    return fig


# ============================================================
# AQI TREND
# ============================================================

def pollution_chart(df):

    if df is None or df.empty:
        return _empty_chart("No AQI data available")

    df = df.copy()

    if "time" not in df.columns:
        return _empty_chart("Time column not available")

    if "AQI" not in df.columns:
        return _empty_chart("AQI data not available")

    df["AQI"] = pd.to_numeric(
        df["AQI"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["AQI"]
    )

    if df.empty:
        return _empty_chart("No valid AQI values")

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
        hovermode="x unified",
        template="plotly_dark",
        height=400
    )

    return fig


# ============================================================
# POLLUTANT BAR CHART
# ============================================================

def pollutant_bar_chart(data):

    pollutants = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3"
    ]

    if data is None:
        return _empty_chart(
            "No pollutant data available"
        )

    # ---------------------------------------------
    # Convert input to dictionary
    # ---------------------------------------------

    if isinstance(data, pd.Series):

        data = data.to_dict()

    elif isinstance(data, pd.DataFrame):

        if data.empty:
            return _empty_chart(
                "No pollutant data available"
            )

        data = data.iloc[0].to_dict()

    elif not isinstance(data, dict):

        try:
            data = dict(data)
        except Exception:
            return _empty_chart(
                "Invalid pollutant data"
            )

    # ---------------------------------------------
    # Extract values safely
    # ---------------------------------------------

    values = []

    for pollutant in pollutants:

        value = data.get(
            pollutant,
            0
        )

        value = _safe_number(
            value,
            0
        )

        values.append(value)

    # ---------------------------------------------
    # Create dataframe
    # ---------------------------------------------

    chart_df = pd.DataFrame({

        "Pollutant": pollutants,

        "Value": values

    })

    # ---------------------------------------------
    # Check whether data actually exists
    # ---------------------------------------------

    if chart_df["Value"].sum() == 0:

        return _empty_chart(
            "No valid pollutant values available"
        )

    # ---------------------------------------------
    # BAR CHART
    # ---------------------------------------------

    fig = px.bar(

        chart_df,

        x="Pollutant",

        y="Value",

        text="Value",

        title="Current Pollutant Levels"

    )

    fig.update_traces(

        texttemplate="%{text:.2f}",

        textposition="outside",

        cliponaxis=False

    )

    fig.update_layout(

        xaxis_title="Pollutant",

        yaxis_title="Concentration",

        template="plotly_dark",

        height=450,

        margin=dict(
            l=40,
            r=30,
            t=70,
            b=50
        ),

        xaxis=dict(
            categoryorder="array",
            categoryarray=pollutants
        )
    )

    return fig


# ============================================================
# AQI GAUGE
# ============================================================

def aqi_gauge(aqi):

    aqi = _safe_number(
        aqi,
        0
    )

    aqi = max(
        0,
        min(
            500,
            aqi
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

                "bar": {
                    "color": "#00c853"
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

                ]
            }
        )
    )

    fig.update_layout(
        height=350,
        template="plotly_dark"
    )

    return fig


# ============================================================
# RISK GAUGE
# ============================================================

def risk_gauge(score):

    score = _safe_number(
        score,
        0
    )

    score = max(
        0,
        min(
            100,
            score
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

                "bar": {
                    "color": "#00c853"
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

                ]
            }
        )
    )

    fig.update_layout(
        height=350,
        template="plotly_dark"
    )

    return fig


# ============================================================
# WEATHER CHART
# ============================================================

def weather_chart(df):

    if df is None or df.empty:
        return _empty_chart(
            "No weather data available"
        )

    if "time" not in df.columns:
        return _empty_chart(
            "Time data not available"
        )

    fig = go.Figure()

    if "Temperature" in df.columns:

        temperature = pd.to_numeric(
            df["Temperature"],
            errors="coerce"
        )

        fig.add_trace(

            go.Scatter(

                x=df["time"],

                y=temperature,

                mode="lines+markers",

                name="Temperature"
            )
        )

    if "Humidity" in df.columns:

        humidity = pd.to_numeric(
            df["Humidity"],
            errors="coerce"
        )

        fig.add_trace(

            go.Scatter(

                x=df["time"],

                y=humidity,

                mode="lines+markers",

                name="Humidity"
            )
        )

    if len(fig.data) == 0:

        return _empty_chart(
            "No weather values available"
        )

    fig.update_layout(

        title="Weather Trend",

        xaxis_title="Time",

        yaxis_title="Value",

        hovermode="x unified",

        template="plotly_dark",

        height=400
    )

    return fig


# ============================================================
# FEATURE CONTRIBUTION
# ============================================================

def contribution_chart(contribution_data):

    if contribution_data is None:
        return _empty_chart(
            "No contribution data available"
        )

    if isinstance(
        contribution_data,
        dict
    ):

        df = pd.DataFrame({

            "Feature":
                list(
                    contribution_data.keys()
                ),

            "Contribution":
                list(
                    contribution_data.values()
                )
        })

    else:

        df = contribution_data.copy()

        if (
            "Contribution (%)"
            in df.columns
        ):

            df = df.rename(

                columns={

                    "Contribution (%)":
                        "Contribution"
                }
            )

    if df.empty:
        return _empty_chart(
            "No contribution data"
        )

    df["Contribution"] = pd.to_numeric(
        df["Contribution"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Contribution"]
    )

    if df.empty:
        return _empty_chart(
            "No valid contribution values"
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

        text="Contribution"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(

        xaxis_title="Contribution (%)",

        yaxis_title="Feature",

        template="plotly_dark",

        height=400
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

    if df is None or df.empty:
        return _empty_chart(
            "No city data available"
        )

    if city_column not in df.columns:
        return _empty_chart(
            "City column not available"
        )

    if value_column not in df.columns:
        return _empty_chart(
            "AQI column not available"
        )

    df = df.copy()

    df[value_column] = pd.to_numeric(
        df[value_column],
        errors="coerce"
    )

    df = df.dropna(
        subset=[value_column]
    )

    if df.empty:
        return _empty_chart(
            "No valid city AQI values"
        )

    fig = px.bar(

        df,

        x=city_column,

        y=value_column,

        title="City AQI Comparison",

        text_auto=".1f"
    )

    fig.update_layout(

        xaxis_title="City",

        yaxis_title="AQI",

        template="plotly_dark"
    )

    return fig


# ============================================================
# POLLUTION COMPARISON
# ============================================================

def pollutant_comparison_chart(
    df,
    city_column="City"
):

    if df is None or df.empty:
        return _empty_chart(
            "No pollution comparison data"
        )

    if city_column not in df.columns:
        return _empty_chart(
            "City column not available"
        )

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
        return _empty_chart(
            "No pollutant columns available"
        )

    chart_df = df.copy()

    for column in available:

        chart_df[column] = pd.to_numeric(
            chart_df[column],
            errors="coerce"
        )

    melted = chart_df.melt(

        id_vars=[city_column],

        value_vars=available,

        var_name="Pollutant",

        value_name="Value"
    )

    melted = melted.dropna(
        subset=["Value"]
    )

    if melted.empty:
        return _empty_chart(
            "No valid pollutant values"
        )

    fig = px.bar(

        melted,

        x=city_column,

        y="Value",

        color="Pollutant",

        barmode="group",

        title="Pollutant Comparison"
    )

    fig.update_layout(
        template="plotly_dark",
        height=450
    )

    return fig


# ============================================================
# MAP
# ============================================================

def city_map(df):

    if df is None or df.empty:
        return _empty_chart(
            "No map data available"
        )

    required = [

        "Latitude",
        "Longitude",
        "City"

    ]

    if not all(
        column in df.columns
        for column in required
    ):

        return _empty_chart(
            "Latitude/Longitude data unavailable"
        )

    map_df = df.copy()

    map_df["Latitude"] = pd.to_numeric(
        map_df["Latitude"],
        errors="coerce"
    )

    map_df["Longitude"] = pd.to_numeric(
        map_df["Longitude"],
        errors="coerce"
    )

    map_df = map_df.dropna(
        subset=[
            "Latitude",
            "Longitude"
        ]
    )

    if map_df.empty:
        return _empty_chart(
            "No valid location data"
        )

    hover_columns = [

        column

        for column in [
            "AQI",
            "PM2.5",
            "PM10"
        ]

        if column in map_df.columns

    ]

    fig = px.scatter_map(

        map_df,

        lat="Latitude",

        lon="Longitude",

        hover_name="City",

        hover_data=hover_columns,

        zoom=4,

        height=500,

        title="Environmental Monitoring Map"
    )

    fig.update_layout(
        map_style="open-street-map",
        template="plotly_dark"
    )

    return fig


# ============================================================
# HISTORY CHART
# ============================================================

def pollutant_history_chart(
    df,
    pollutants=None
):

    if df is None or df.empty:
        return _empty_chart(
            "No pollutant history available"
        )

    if "time" not in df.columns:
        return _empty_chart(
            "Time column unavailable"
        )

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

    if not available:

        return _empty_chart(
            "No pollutant history columns"
        )

    fig = go.Figure()

    for pollutant in available:

        values = pd.to_numeric(

            df[pollutant],

            errors="coerce"

        )

        fig.add_trace(

            go.Scatter(

                x=df["time"],

                y=values,

                mode="lines",

                name=pollutant
            )
        )

    if len(fig.data) == 0:

        return _empty_chart(
            "No valid historical values"
        )

    fig.update_layout(

        title="Pollutant History",

        xaxis_title="Time",

        yaxis_title="Concentration",

        hovermode="x unified",

        template="plotly_dark",

        height=450
    )

    return fig
