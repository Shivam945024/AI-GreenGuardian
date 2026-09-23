import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data_loader import (
    load_pollution_dataset,
    load_weather_dataset,
    load_city_dataset,
    get_city_data,
)

from src.pollution_api import (
    build_environment_record,
    air_quality_history,
    get_combined_environment_data,
)

from src.aqi_calculator import (
    calculate_aqi,
    aqi_category,
    aqi_description,
)

from src.risk_engine import (
    environmental_risk,
    risk_level,
    risk_message,
)

from src.prediction import (
    predict_next_aqi,
    prediction_change,
    predict_pollution,
    predict_risk,
)

from src.recommendations import (
    recommendations,
    short_recommendation,
)

from src.explainability import (
    contribution_dataframe,
    top_contributors,
    explain_prediction,
)

from utils.alerts import (
    get_all_alerts,
    has_critical_alerts,
    alert_summary,
)

from utils.charts import (
    pollution_chart,
    pollutant_bar_chart,
    aqi_gauge,
    risk_gauge,
    weather_chart,
    contribution_chart,
    city_comparison_chart,
    pollutant_comparison_chart,
    city_map,
)

from utils.helpers import (
    dataframe_to_csv,
    clean_environment_data,
    current_timestamp,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI GreenGuardian",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(34,197,94,0.10), transparent 25%),
        radial-gradient(circle at 90% 10%, rgba(14,165,233,0.08), transparent 25%),
        #07110d;
    color: #f8fafc;
}

/* Main container */

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #07110d 0%,
            #0b1812 50%,
            #07110d 100%
        );
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Hero */

.hero {
    padding: 30px;
    border-radius: 28px;
    margin-bottom: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(22,163,74,0.20),
            rgba(14,165,233,0.10)
        );

    border: 1px solid rgba(74,222,128,0.18);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.30);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;

    background:
        linear-gradient(
            90deg,
            #4ade80,
            #22c55e,
            #38bdf8
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #a7f3d0;
    font-size: 16px;
    line-height: 1.7;
}

/* Cards */

.metric-card {
    padding: 22px;
    min-height: 145px;

    border-radius: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.065),
            rgba(255,255,255,0.025)
        );

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 15px 40px rgba(0,0,0,0.22);
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #f8fafc;
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-description {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 4px;
}

/* Section heading */

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #f8fafc;
    margin-top: 30px;
    margin-bottom: 15px;
}

.section-subtitle {
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Status */

.status-good {
    color: #4ade80;
}

.status-warning {
    color: #facc15;
}

.status-danger {
    color: #fb7185;
}

/* Alert */

.alert-box {
    padding: 17px 20px;
    border-radius: 16px;
    margin: 8px 0;

    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(248,113,113,0.20);
}

.alert-title {
    font-weight: 700;
    color: #fca5a5;
}

.alert-message {
    color: #cbd5e1;
    font-size: 14px;
}

/* Recommendation */

.recommendation {
    padding: 17px 20px;
    margin: 8px 0;

    border-radius: 16px;

    background: rgba(34,197,94,0.07);
    border: 1px solid rgba(74,222,128,0.14);
}

.recommendation-number {
    color: #4ade80;
    font-weight: 800;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 40px;
    padding-bottom: 10px;
}

/* Buttons */

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(74,222,128,0.25);
    background: rgba(34,197,94,0.10);
    color: #bbf7d0;
    font-weight: 600;
}

.stButton > button:hover {
    border-color: #4ade80;
    background: rgba(34,197,94,0.18);
}

/* Selectbox */

div[data-baseweb="select"] > div {
    border-radius: 12px;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

/* Hide Streamlit */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CACHE DATA
# ============================================================

@st.cache_data(ttl=3600)
def load_city_data_cached():
    return load_city_dataset()


@st.cache_data(ttl=1800)
def load_pollution_data_cached():
    return load_pollution_dataset()


@st.cache_data(ttl=1800)
def load_weather_data_cached():
    return load_weather_dataset()


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_number(value, default=0.0):
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def get_first_value(data, keys, default=0.0):

    for key in keys:

        if key in data:

            value = safe_number(data[key], None)

            if value is not None:
                return value

    return default


def build_fallback_environment(city):

    pollution_df = load_pollution_data_cached()

    if pollution_df.empty:
        return {
            "PM2.5": 35,
            "PM10": 65,
            "NO2": 25,
            "SO2": 10,
            "CO": 0.8,
            "O3": 45,
            "Temperature": 28,
            "Humidity": 55,
            "Wind Speed": 8,
        }

    row = pollution_df.iloc[-1]

    result = {}

    for column in [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "Temperature",
        "Humidity",
        "Wind Speed",
    ]:

        if column in row:
            result[column] = safe_number(row[column])

    defaults = {
        "PM2.5": 35,
        "PM10": 65,
        "NO2": 25,
        "SO2": 10,
        "CO": 0.8,
        "O3": 45,
        "Temperature": 28,
        "Humidity": 55,
        "Wind Speed": 8,
    }

    for key, value in defaults.items():

        if key not in result:
            result[key] = value

    return result


def get_environment(city, live_mode):

    if not live_mode:
        return build_fallback_environment(city), "Dataset Demo"

    try:

        city_info = get_city_data(city)

        if city_info is None:
            return build_fallback_environment(city), "Dataset Demo"

        latitude = safe_number(
            city_info.get("Latitude", 0)
        )

        longitude = safe_number(
            city_info.get("Longitude", 0)
        )

        if latitude == 0 and longitude == 0:
            return build_fallback_environment(city), "Dataset Demo"

        data = build_environment_record(
            latitude,
            longitude
        )

        if data:
            return clean_environment_data(data), "Live API"

    except Exception as error:

        st.session_state["api_error"] = str(error)

    return build_fallback_environment(city), "Dataset Demo"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:15px 0 20px 0;
        ">
            <div style="font-size:48px;">🌱</div>
            <div style="
                font-size:22px;
                font-weight:800;
                color:#4ade80;
            ">
                GreenGuardian
            </div>
            <div style="
                color:#64748b;
                font-size:12px;
            ">
                AI Environmental Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🌫 Pollution Monitor",
            "🤖 AI Prediction",
            "⚠️ Risk Intelligence",
            "🏙 City Comparison",
            "🧠 Explainable AI",
            "💡 Recommendations",
            "🚨 Alerts",
            "📥 Data Export",
        ],
    )

    st.divider()

    st.markdown("### 📍 Monitoring")

    city_df = load_city_data_cached()

    if city_df.empty or "City" not in city_df.columns:

        cities = [
            "Lucknow",
            "Delhi",
            "Mumbai",
            "Bengaluru",
            "Kanpur",
            "Kolkata",
            "Chennai",
            "Hyderabad",
            "Pune",
            "Jaipur",
        ]

    else:

        cities = city_df["City"].dropna().astype(str).tolist()

    selected_city = st.selectbox(
        "Select City",
        cities,
        index=0,
    )

    live_mode = st.toggle(
        "🌐 Live Environmental Data",
        value=False,
    )

    if st.button(
        "🔄 Refresh Data",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.markdown(
        """
        **System**

        🟢 AI Engine: Active  
        🟢 Dashboard: Online  
        🟢 Monitoring: Ready
        """
    )


# ============================================================
# LOAD ENVIRONMENT DATA
# ============================================================

environment, data_source = get_environment(
    selected_city,
    live_mode,
)

environment = clean_environment_data(environment)


# ============================================================
# CALCULATE AQI
# ============================================================

aqi = calculate_aqi(
    environment.get("PM2.5", 0),
    environment.get("PM10", 0),
    environment.get("NO2", 0),
    environment.get("SO2", 0),
    environment.get("CO", 0),
    environment.get("O3", 0),
)

aqi = safe_number(aqi)


# ============================================================
# RISK
# ============================================================

try:

    risk_score = environmental_risk(
        environment,
        aqi
    )

except Exception:

    risk_score = 0

risk_score = safe_number(risk_score)

try:
    risk_status = risk_level(risk_score)
except Exception:
    risk_status = "LOW"


# ============================================================
# PREDICTION
# ============================================================

try:

    predicted_aqi = predict_next_aqi(
        environment,
        current_aqi=aqi
    )

except Exception:

    predicted_aqi = aqi

predicted_aqi = safe_number(
    predicted_aqi,
    aqi
)

change = prediction_change(
    aqi,
    predicted_aqi
)


# ============================================================
# GLOBAL HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="hero-title">
            🌱 AI GreenGuardian
        </div>

        <div class="hero-subtitle">
            Intelligent Environmental Monitoring,
            Pollution Prediction & Risk Intelligence Platform
        </div>

        <div style="
            margin-top:18px;
            display:flex;
            gap:10px;
            flex-wrap:wrap;
        ">

            <span style="
                padding:7px 13px;
                border-radius:20px;
                background:rgba(34,197,94,0.12);
                color:#86efac;
                font-size:12px;
            ">
                📍 {selected_city}
            </span>

            <span style="
                padding:7px 13px;
                border-radius:20px;
                background:rgba(56,189,248,0.10);
                color:#7dd3fc;
                font-size:12px;
            ">
                🌐 {data_source}
            </span>

            <span style="
                padding:7px 13px;
                border-radius:20px;
                background:rgba(168,85,247,0.10);
                color:#d8b4fe;
                font-size:12px;
            ">
                🤖 AI Powered
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# METRIC CARDS
# ============================================================

metric_cols = st.columns(4)

with metric_cols[0]:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Air Quality Index
            </div>

            <div class="metric-value">
                {aqi:.0f}
            </div>

            <div class="metric-description">
                {aqi_category(aqi)}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_cols[1]:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                PM2.5
            </div>

            <div class="metric-value">
                {environment.get("PM2.5", 0):.1f}
            </div>

            <div class="metric-description">
                μg/m³ concentration
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_cols[2]:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Predicted AQI
            </div>

            <div class="metric-value">
                {predicted_aqi:.0f}
            </div>

            <div class="metric-description">
                {change}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_cols[3]:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Environmental Risk
            </div>

            <div class="metric-value">
                {risk_score:.0f}
            </div>

            <div class="metric-description">
                {risk_status}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">🌍 Environmental Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section-subtitle">
            Real-time environmental intelligence for
            <b>{selected_city}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:

        st.plotly_chart(
            aqi_gauge(aqi),
            use_container_width=True,
        )

    with right:

        st.plotly_chart(
            risk_gauge(risk_score),
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-title">🧪 Pollutant Profile</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        pollutant_bar_chart(environment),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-title">🌡️ Environmental Conditions</div>',
        unsafe_allow_html=True,
    )

    weather_data = pd.DataFrame(
        [
            {
                "Metric": "Temperature",
                "Value": environment.get(
                    "Temperature", 0
                ),
            },
            {
                "Metric": "Humidity",
                "Value": environment.get(
                    "Humidity", 0
                ),
            },
            {
                "Metric": "Wind Speed",
                "Value": environment.get(
                    "Wind Speed", 0
                ),
            },
        ]
    )

    st.dataframe(
        weather_data,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">🚨 Environmental Status</div>',
        unsafe_allow_html=True,
    )

    if aqi <= 50:

        st.success(
            f"🟢 Air quality is currently in the {aqi_category(aqi)} range."
        )

    elif aqi <= 100:

        st.warning(
            f"🟡 Air quality is currently {aqi_category(aqi)}."
        )

    elif aqi <= 200:

        st.warning(
            f"🟠 Elevated pollution detected. Current AQI: {aqi:.0f}"
        )

    else:

        st.error(
            f"🔴 High pollution detected. Current AQI: {aqi:.0f}"
        )


# ============================================================
# POLLUTION MONITOR
# ============================================================

elif page == "🌫 Pollution Monitor":

    st.markdown(
        '<div class="section-title">🌫 Pollution Monitoring Center</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Monitor major air pollutants and environmental
        conditions affecting air quality.
        """,
    )

    pollutant_data = pd.DataFrame(
        {
            "Pollutant": [
                "PM2.5",
                "PM10",
                "NO2",
                "SO2",
                "CO",
                "O3",
            ],
            "Value": [
                environment.get("PM2.5", 0),
                environment.get("PM10", 0),
                environment.get("NO2", 0),
                environment.get("SO2", 0),
                environment.get("CO", 0),
                environment.get("O3", 0),
            ],
        }
    )

    st.dataframe(
        pollutant_data,
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(
        pollutant_bar_chart(environment),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-title">📈 Historical Trend</div>',
        unsafe_allow_html=True,
    )

    try:

        if live_mode:

            city_info = get_city_data(selected_city)

            history = air_quality_history(
                city_info["Latitude"],
                city_info["Longitude"],
            )

            if history is not None and not history.empty:

                st.plotly_chart(
                    pollution_chart(history),
                    use_container_width=True,
                )

            else:

                st.info(
                    "Historical API data is unavailable."
                )

        else:

            pollution_df = load_pollution_data_cached()

            if not pollution_df.empty:

                st.plotly_chart(
                    pollution_chart(pollution_df),
                    use_container_width=True,
                )

            else:

                st.info(
                    "No historical dataset available."
                )

    except Exception as error:

        st.warning(
            f"Unable to load historical data: {error}"
        )


# ============================================================
# AI PREDICTION
# ============================================================

elif page == "🤖 AI Prediction":

    st.markdown(
        '<div class="section-title">🤖 AI Pollution Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Machine-learning models estimate future environmental
        conditions using pollutant and weather features.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current AQI",
            f"{aqi:.0f}",
        )

    with col2:

        st.metric(
            "Predicted AQI",
            f"{predicted_aqi:.0f}",
            delta=f"{predicted_aqi - aqi:.1f}",
        )

    with col3:

        st.metric(
            "Change",
            change,
        )

    st.markdown(
        '<div class="section-title">📊 Prediction Analysis</div>',
        unsafe_allow_html=True,
    )

    prediction_df = pd.DataFrame(
        {
            "Metric": [
                "Current AQI",
                "Predicted AQI",
            ],
            "Value": [
                aqi,
                predicted_aqi,
            ],
        }
    )

    st.bar_chart(
        prediction_df.set_index("Metric")
    )

    try:

        pollution_prediction = predict_pollution(
            environment
        )

        st.markdown(
            '<div class="section-title">🧪 Model Output</div>',
            unsafe_allow_html=True,
        )

        st.json(
            pollution_prediction
            if isinstance(
                pollution_prediction,
                dict
            )
            else {
                "Predicted AQI":
                pollution_prediction
            }
        )

    except Exception:

        st.info(
            "Pollution prediction model output is unavailable."
        )

    st.info(
        "⚠️ AI predictions are estimates and should not be treated "
        "as official environmental forecasts."
    )


# ============================================================
# RISK INTELLIGENCE
# ============================================================

elif page == "⚠️ Risk Intelligence":

    st.markdown(
        '<div class="section-title">⚠️ Environmental Risk Intelligence</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:

        st.plotly_chart(
            risk_gauge(risk_score),
            use_container_width=True,
        )

    with right:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Current Risk Level
                </div>

                <div class="metric-value">
                    {risk_status}
                </div>

                <div class="metric-description">
                    Risk Score: {risk_score:.1f}/100
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        try:

            message = risk_message(
                risk_status
            )

        except Exception:

            message = "Monitor environmental conditions."

        st.write("")
        st.info(message)

    st.markdown(
        '<div class="section-title">📊 Risk Components</div>',
        unsafe_allow_html=True,
    )

    risk_components = pd.DataFrame(
        {
            "Component": [
                "AQI",
                "PM2.5",
                "PM10",
                "NO2",
                "Temperature",
                "Humidity",
                "Wind Speed",
            ],
            "Value": [
                aqi,
                environment.get("PM2.5", 0),
                environment.get("PM10", 0),
                environment.get("NO2", 0),
                environment.get("Temperature", 0),
                environment.get("Humidity", 0),
                environment.get("Wind Speed", 0),
            ],
        }
    )

    st.dataframe(
        risk_components,
        use_container_width=True,
        hide_index=True,
    )

    try:

        risk_prediction = predict_risk(
            environment
        )

        st.markdown(
            '<div class="section-title">🤖 AI Risk Prediction</div>',
            unsafe_allow_html=True,
        )

        st.write(risk_prediction)

    except Exception:

        pass


# ============================================================
# CITY COMPARISON
# ============================================================

elif page == "🏙 City Comparison":

    st.markdown(
        '<div class="section-title">🏙️ Multi-City Environmental Intelligence</div>',
        unsafe_allow_html=True,
    )

    pollution_df = load_pollution_data_cached()

    if pollution_df.empty:

        st.warning(
            "Pollution dataset is empty."
        )

    else:

        if "City" not in pollution_df.columns:

            st.info(
                "Add a City column to pollution_dataset.csv "
                "to enable city-level comparison."
            )

        else:

            city_summary = (
                pollution_df
                .groupby("City", as_index=False)
                .agg(
                    {
                        "AQI": "mean",
                        "PM2.5": "mean",
                        "PM10": "mean",
                        "NO2": "mean",
                    }
                )
            )

            city_summary = city_summary.sort_values(
                "AQI",
                ascending=False,
            )

            st.dataframe(
                city_summary,
                use_container_width=True,
                hide_index=True,
            )

            st.plotly_chart(
                city_comparison_chart(
                    city_summary,
                    city_column="City",
                    value_column="AQI",
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                pollutant_comparison_chart(
                    city_summary,
                    city_column="City",
                ),
                use_container_width=True,
            )

            if (
                "Latitude" in city_df.columns
                and "Longitude" in city_df.columns
            ):

                map_data = city_df.copy()

                if "AQI" not in map_data.columns:

                    map_data["AQI"] = 0

                st.plotly_chart(
                    city_map(map_data),
                    use_container_width=True,
                )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    st.markdown(
        '<div class="section-title">🧠 Explainable Environmental AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Understand which environmental factors are contributing
        most to the current pollution/risk assessment.
        """
    )

    try:

        contribution_df = contribution_dataframe(
            environment
        )

        if contribution_df is not None:

            st.plotly_chart(
                contribution_chart(
                    contribution_df
                ),
                use_container_width=True,
            )

            st.markdown(
                '<div class="section-title">🔎 Top Contributors</div>',
                unsafe_allow_html=True,
            )

            contributors = top_contributors(
                environment,
                n=5,
            )

            if contributors:

                for index, item in enumerate(
                    contributors,
                    start=1
                ):

                    st.markdown(
                        f"""
                        <div class="recommendation">

                            <span class="recommendation-number">
                                #{index}
                            </span>

                            &nbsp;&nbsp;

                            {item}

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            explanation = explain_prediction(
                environment,
                predicted_aqi,
            )

            if explanation:

                st.markdown(
                    '<div class="section-title">💬 AI Explanation</div>',
                    unsafe_allow_html=True,
                )

                st.info(
                    explanation
                )

    except Exception as error:

        st.warning(
            f"Explainability module unavailable: {error}"
        )

    st.caption(
        "Note: current explainability is a feature-contribution "
        "indicator, not causal inference or SHAP-based explanation."
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.markdown(
        '<div class="section-title">💡 Smart Environmental Recommendations</div>',
        unsafe_allow_html=True,
    )

    try:

        recs = recommendations(
            aqi,
            risk_score,
            environment,
        )

    except Exception:

        recs = []

    if not recs:

        st.success(
            "🌱 No major environmental intervention is currently recommended."
        )

    else:

        for index, rec in enumerate(
            recs,
            start=1
        ):

            st.markdown(
                f"""
                <div class="recommendation">

                    <span class="recommendation-number">
                        {index:02d}
                    </span>

                    &nbsp;&nbsp;

                    {rec}

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">⚡ Quick Guidance</div>',
        unsafe_allow_html=True,
    )

    try:

        st.success(
            short_recommendation(
                aqi,
                risk_score,
            )
        )

    except Exception:

        st.info(
            "Monitor AQI and local environmental guidance."
        )


# ============================================================
# ALERTS
# ============================================================

elif page == "🚨 Alerts":

    st.markdown(
        '<div class="section-title">🚨 Environmental Alert Center</div>',
        unsafe_allow_html=True,
    )

    try:

        alerts = get_all_alerts(
            aqi,
            risk_score,
            environment,
        )

    except Exception:

        alerts = []

    if not alerts:

        st.success(
            "🟢 No active environmental alerts."
        )

    else:

        for alert in alerts:

            if isinstance(alert, dict):

                title = alert.get(
                    "title",
                    "Environmental Alert"
                )

                message = alert.get(
                    "message",
                    ""
                )

            else:

                title = "Environmental Alert"
                message = str(alert)

            st.markdown(
                f"""
                <div class="alert-box">

                    <div class="alert-title">
                        🚨 {title}
                    </div>

                    <div class="alert-message">
                        {message}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">📋 Alert Summary</div>',
        unsafe_allow_html=True,
    )

    try:

        st.write(
            alert_summary(
                aqi,
                risk_score,
                environment,
            )
        )

    except Exception:

        pass

    if has_critical_alerts(
        aqi,
        risk_score,
    ):

        st.error(
            "Critical environmental conditions detected. "
            "Follow applicable local health/environment guidance."
        )


# ============================================================
# DATA EXPORT
# ============================================================

elif page == "📥 Data Export":

    st.markdown(
        '<div class="section-title">📥 Environmental Data Export</div>',
        unsafe_allow_html=True,
    )

    export_data = {
        "City": selected_city,
        "Data Source": data_source,
        "Timestamp": current_timestamp(),
        "AQI": aqi,
        "Predicted AQI": predicted_aqi,
        "Risk Score": risk_score,
        "Risk Level": risk_status,
        "PM2.5": environment.get("PM2.5", 0),
        "PM10": environment.get("PM10", 0),
        "NO2": environment.get("NO2", 0),
        "SO2": environment.get("SO2", 0),
        "CO": environment.get("CO", 0),
        "O3": environment.get("O3", 0),
        "Temperature": environment.get(
            "Temperature",
            0,
        ),
        "Humidity": environment.get(
            "Humidity",
            0,
        ),
        "Wind Speed": environment.get(
            "Wind Speed",
            0,
        ),
    }

    export_df = pd.DataFrame(
        [export_data]
    )

    st.dataframe(
        export_df,
        use_container_width=True,
        hide_index=True,
    )

    try:

        csv_data = dataframe_to_csv(
            export_df
        )

    except Exception:

        csv_data = export_df.to_csv(
            index=False
        )

    st.download_button(
        label="⬇️ Download Environmental Report",
        data=csv_data,
        file_name=(
            f"greenguardian_"
            f"{selected_city.lower()}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🌱 <b>AI GreenGuardian</b>
        &nbsp;•&nbsp;
        Smart Pollution & Environmental Risk System

        <br><br>

        Built with Python • Streamlit • Machine Learning •
        Open-Meteo • Plotly

        <br><br>

        ⚠️ Prototype environmental intelligence system.
        AQI/risk outputs should not be treated as official
        regulatory measurements or medical advice.

    </div>
    """,
    unsafe_allow_html=True,
)
