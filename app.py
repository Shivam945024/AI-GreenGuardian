import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
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

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(34,197,94,0.10),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(14,165,233,0.08),
                transparent 25%
            ),
            #07110d;

        color: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ================= SIDEBAR ================= */

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

    /* ================= HERO ================= */

    .hero {
        padding: 32px;
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

    /* ================= BADGES ================= */

    .badge-container {
        margin-top: 18px;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }

    .badge {
        padding: 7px 13px;
        border-radius: 20px;
        font-size: 12px;
    }

    .badge-location {
        background: rgba(34,197,94,0.12);
        color: #86efac;
    }

    .badge-source {
        background: rgba(56,189,248,0.10);
        color: #7dd3fc;
    }

    .badge-ai {
        background: rgba(168,85,247,0.10);
        color: #d8b4fe;
    }

    /* ================= METRIC CARDS ================= */

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

    /* ================= SECTION ================= */

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

    /* ================= ALERT ================= */

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
        margin-top: 5px;
    }

    /* ================= RECOMMENDATION ================= */

    .recommendation {
        padding: 17px 20px;
        margin: 8px 0;

        border-radius: 16px;

        background: rgba(34,197,94,0.07);
        border: 1px solid rgba(74,222,128,0.14);

        color: #d1fae5;
    }

    .recommendation-number {
        color: #4ade80;
        font-weight: 800;
    }

    /* ================= FOOTER ================= */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding-top: 40px;
        padding-bottom: 10px;
        line-height: 1.7;
    }

    /* ================= BUTTONS ================= */

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

    /* ================= DATAFRAME ================= */

    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    /* ================= HIDE STREAMLIT ================= */

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
def get_city_dataset():
    try:
        return load_city_dataset()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_pollution_dataset():
    try:
        return load_pollution_dataset()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_weather_dataset():
    try:
        return load_weather_dataset()
    except Exception:
        return pd.DataFrame()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def fallback_environment():

    pollution_df = get_pollution_dataset()

    defaults = {
        "PM2.5": 35.0,
        "PM10": 65.0,
        "NO2": 25.0,
        "SO2": 10.0,
        "CO": 0.8,
        "O3": 45.0,
        "Temperature": 28.0,
        "Humidity": 55.0,
        "Wind Speed": 8.0,
    }

    if pollution_df.empty:
        return defaults

    row = pollution_df.iloc[-1]

    for key in defaults:

        if key in row:

            try:

                value = float(row[key])

                if pd.notna(value):
                    defaults[key] = value

            except Exception:
                pass

    return defaults


def get_environment_data(city, live):

    if not live:
        return fallback_environment(), "Dataset Demo"

    try:

        city_info = get_city_data(city)

        if city_info is None:
            return fallback_environment(), "Dataset Demo"

        latitude = safe_float(
            city_info.get("Latitude", 0)
        )

        longitude = safe_float(
            city_info.get("Longitude", 0)
        )

        if latitude == 0 or longitude == 0:

            return (
                fallback_environment(),
                "Dataset Demo"
            )

        data = build_environment_record(
            latitude,
            longitude,
        )

        if data:

            return (
                clean_environment_data(data),
                "Live API",
            )

    except Exception as error:

        st.session_state["api_error"] = str(error)

    return fallback_environment(), "Dataset Demo"


def get_aqi_category_safe(aqi):

    try:
        return aqi_category(aqi)
    except Exception:

        if aqi <= 50:
            return "Good"

        if aqi <= 100:
            return "Moderate"

        if aqi <= 200:
            return "Poor"

        if aqi <= 300:
            return "Very Poor"

        return "Severe"


def calculate_current_aqi(environment):

    try:

        return float(
            calculate_aqi(
                environment.get("PM2.5", 0),
                environment.get("PM10", 0),
                environment.get("NO2", 0),
                environment.get("SO2", 0),
                environment.get("CO", 0),
                environment.get("O3", 0),
            )
        )

    except Exception:

        return 0.0


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

            <div style="
                font-size:48px;
            ">
                🌱
            </div>

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

    st.markdown(
        "### 🧭 Navigation"
    )

    page = st.radio(
        "Select Module",
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
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        "### 📍 Monitoring Location"
    )

    city_df = get_city_dataset()

    if (
        not city_df.empty
        and "City" in city_df.columns
    ):

        cities = (
            city_df["City"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

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

    selected_city = st.selectbox(
        "City",
        cities,
    )

    live_mode = st.toggle(
        "🌐 Use Live Data",
        value=False,
    )

    st.divider()

    if st.button(
        "🔄 Refresh Dashboard",
        use_container_width=True,
    ):

        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.markdown(
        """
        <div style="
            padding:15px;
            border-radius:15px;
            background:rgba(34,197,94,0.06);
            border:1px solid rgba(74,222,128,0.10);
        ">

            <div style="
                color:#4ade80;
                font-weight:700;
            ">
                🟢 System Status
            </div>

            <br>

            🟢 Dashboard Online<br>
            🟢 AI Engine Ready<br>
            🟢 Monitoring Active

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GET CURRENT ENVIRONMENT
# ============================================================

environment, data_source = get_environment_data(
    selected_city,
    live_mode,
)

environment = clean_environment_data(
    environment
)


# ============================================================
# AQI
# ============================================================

aqi = calculate_current_aqi(
    environment
)

aqi_status = get_aqi_category_safe(aqi)


# ============================================================
# RISK
# ============================================================

try:

    risk_score = float(
        environmental_risk(
            environment,
            aqi,
        )
    )

except Exception:

    risk_score = 0.0

try:

    risk_status = risk_level(
        risk_score
    )

except Exception:

    if risk_score < 25:
        risk_status = "LOW"

    elif risk_score < 50:
        risk_status = "MODERATE"

    elif risk_score < 75:
        risk_status = "HIGH"

    else:
        risk_status = "CRITICAL"


# ============================================================
# AI PREDICTION
# ============================================================

try:

    predicted_aqi = float(
        predict_next_aqi(
            environment,
            current_aqi=aqi,
        )
    )

except Exception:

    predicted_aqi = aqi

try:

    change_text = prediction_change(
        aqi,
        predicted_aqi,
    )

except Exception:

    difference = predicted_aqi - aqi

    if difference > 0:
        change_text = f"↑ {difference:.1f}"

    elif difference < 0:
        change_text = f"↓ {abs(difference):.1f}"

    else:
        change_text = "Stable"


# ============================================================
# HERO
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

        <div class="badge-container">

            <span class="badge badge-location">
                📍 {selected_city}
            </span>

            <span class="badge badge-source">
                🌐 {data_source}
            </span>

            <span class="badge badge-ai">
                🤖 AI Powered
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

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
                {aqi_status}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with c2:

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


with c3:

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
                {change_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with c4:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Risk Score
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
            Current environmental intelligence for
            <b>{selected_city}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        try:

            st.plotly_chart(
                aqi_gauge(aqi),
                use_container_width=True,
            )

        except Exception:

            st.metric(
                "AQI",
                f"{aqi:.0f}"
            )

    with col2:

        try:

            st.plotly_chart(
                risk_gauge(risk_score),
                use_container_width=True,
            )

        except Exception:

            st.metric(
                "Risk",
                f"{risk_score:.0f}"
            )

    st.markdown(
        '<div class="section-title">🧪 Pollutant Profile</div>',
        unsafe_allow_html=True,
    )

    try:

        st.plotly_chart(
            pollutant_bar_chart(environment),
            use_container_width=True,
        )

    except Exception:

        pollutant_df = pd.DataFrame(
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
            pollutant_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        '<div class="section-title">🌡️ Environment Conditions</div>',
        unsafe_allow_html=True,
    )

    weather_df = pd.DataFrame(
        {
            "Metric": [
                "Temperature",
                "Humidity",
                "Wind Speed",
            ],
            "Value": [
                environment.get(
                    "Temperature",
                    0,
                ),
                environment.get(
                    "Humidity",
                    0,
                ),
                environment.get(
                    "Wind Speed",
                    0,
                ),
            ],
        }
    )

    st.dataframe(
        weather_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">🚦 Current Status</div>',
        unsafe_allow_html=True,
    )

    if aqi <= 50:

        st.success(
            f"🟢 Air quality is currently {aqi_status}."
        )

    elif aqi <= 100:

        st.warning(
            f"🟡 Air quality is currently {aqi_status}."
        )

    elif aqi <= 200:

        st.warning(
            f"🟠 Elevated pollution detected. AQI: {aqi:.0f}"
        )

    else:

        st.error(
            f"🔴 High pollution detected. AQI: {aqi:.0f}"
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
        <div class="section-subtitle">
            Monitor major pollutants and environmental
            conditions affecting air quality.
        </div>
        """,
        unsafe_allow_html=True,
    )

    pollutant_df = pd.DataFrame(
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
        pollutant_df,
        use_container_width=True,
        hide_index=True,
    )

    try:

        st.plotly_chart(
            pollutant_bar_chart(environment),
            use_container_width=True,
        )

    except Exception:
        pass

    st.markdown(
        '<div class="section-title">📈 Pollution History</div>',
        unsafe_allow_html=True,
    )

    try:

        if live_mode:

            city_info = get_city_data(
                selected_city
            )

            history = air_quality_history(
                city_info["Latitude"],
                city_info["Longitude"],
            )

            if (
                history is not None
                and not history.empty
            ):

                st.plotly_chart(
                    pollution_chart(history),
                    use_container_width=True,
                )

            else:

                st.info(
                    "Live historical pollution data is unavailable."
                )

        else:

            pollution_df = get_pollution_dataset()

            if not pollution_df.empty:

                st.plotly_chart(
                    pollution_chart(
                        pollution_df
                    ),
                    use_container_width=True,
                )

            else:

                st.info(
                    "No pollution history dataset found."
                )

    except Exception as error:

        st.warning(
            f"Historical data unavailable: {error}"
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
        <div class="section-subtitle">
            Machine-learning models analyze pollution and
            environmental variables to estimate future AQI.
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "Current AQI",
            f"{aqi:.0f}",
        )

    with p2:

        st.metric(
            "Predicted AQI",
            f"{predicted_aqi:.0f}",
            delta=f"{predicted_aqi - aqi:.1f}",
        )

    with p3:

        st.metric(
            "Expected Change",
            change_text,
        )

    st.markdown(
        '<div class="section-title">📊 Prediction Comparison</div>',
        unsafe_allow_html=True,
    )

    prediction_df = pd.DataFrame(
        {
            "AQI": [
                aqi,
                predicted_aqi,
            ]
        },
        index=[
            "Current",
            "Predicted",
        ],
    )

    st.bar_chart(
        prediction_df
    )

    st.markdown(
        '<div class="section-title">🧠 Model Output</div>',
        unsafe_allow_html=True,
    )

    try:

        model_prediction = predict_pollution(
            environment
        )

        if isinstance(
            model_prediction,
            dict,
        ):

            st.json(
                model_prediction
            )

        else:

            st.metric(
                "Model Prediction",
                str(model_prediction),
            )

    except Exception as error:

        st.info(
            f"Pollution model output unavailable: {error}"
        )

    st.warning(
        "⚠️ AI predictions are estimates and should not "
        "be treated as official environmental forecasts."
    )


# ============================================================
# RISK INTELLIGENCE
# ============================================================

elif page == "⚠️ Risk Intelligence":

    st.markdown(
        '<div class="section-title">⚠️ Environmental Risk Intelligence</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        try:

            st.plotly_chart(
                risk_gauge(risk_score),
                use_container_width=True,
            )

        except Exception:

            st.metric(
                "Risk Score",
                f"{risk_score:.1f}/100",
            )

    with col2:

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

            st.info(
                risk_message(
                    risk_status
                )
            )

        except Exception:
            pass

    st.markdown(
        '<div class="section-title">📊 Risk Factors</div>',
        unsafe_allow_html=True,
    )

    risk_df = pd.DataFrame(
        {
            "Factor": [
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
        risk_df,
        use_container_width=True,
        hide_index=True,
    )

    try:

        ai_risk = predict_risk(
            environment
        )

        st.markdown(
            '<div class="section-title">🤖 AI Risk Model</div>',
            unsafe_allow_html=True,
        )

        st.write(
            ai_risk
        )

    except Exception:
        pass


# ============================================================
# CITY COMPARISON
# ============================================================

elif page == "🏙 City Comparison":

    st.markdown(
        '<div class="section-title">🏙️ City Environmental Comparison</div>',
        unsafe_allow_html=True,
    )

    pollution_df = get_pollution_dataset()

    if pollution_df.empty:

        st.warning(
            "pollution_dataset.csv is empty or unavailable."
        )

    elif "City" not in pollution_df.columns:

        st.info(
            """
            Your current pollution dataset does not contain
            a `City` column.

            Add a City column to enable city-level comparison.
            """
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

        try:

            st.plotly_chart(
                city_comparison_chart(
                    city_summary,
                    city_column="City",
                    value_column="AQI",
                ),
                use_container_width=True,
            )

        except Exception:
            pass

        try:

            st.plotly_chart(
                pollutant_comparison_chart(
                    city_summary,
                    city_column="City",
                ),
                use_container_width=True,
            )

        except Exception:
            pass

        if (
            not city_df.empty
            and "Latitude" in city_df.columns
            and "Longitude" in city_df.columns
        ):

            map_data = city_df.copy()

            if "AQI" not in map_data.columns:
                map_data["AQI"] = 0

            try:

                st.markdown(
                    '<div class="section-title">🗺️ Environmental Map</div>',
                    unsafe_allow_html=True,
                )

                st.plotly_chart(
                    city_map(map_data),
                    use_container_width=True,
                )

            except Exception:
                pass


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
        Understand the environmental factors contributing
        to the current pollution and risk assessment.
        """
    )

    try:

        contribution_df = contribution_dataframe(
            environment
        )

        if (
            contribution_df is not None
            and not contribution_df.empty
        ):

            st.plotly_chart(
                contribution_chart(
                    contribution_df
                ),
                use_container_width=True,
            )

    except Exception as error:

        st.warning(
            f"Contribution analysis unavailable: {error}"
        )

    st.markdown(
        '<div class="section-title">🔎 Top Environmental Contributors</div>',
        unsafe_allow_html=True,
    )

    try:

        contributors = top_contributors(
            environment,
            n=5,
        )

        if contributors:

            for i, contributor in enumerate(
                contributors,
                start=1,
            ):

                st.markdown(
                    f"""
                    <div class="recommendation">

                        <span class="recommendation-number">
                            #{i}
                        </span>

                        &nbsp;&nbsp;

                        {contributor}

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except Exception as error:

        st.info(
            f"Contributor analysis unavailable: {error}"
        )

    try:

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

    except Exception:
        pass

    st.caption(
        "Note: this version uses feature-contribution indicators. "
        "They are not causal explanations or SHAP values."
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
            "🌱 No major environmental intervention "
            "is currently recommended."
        )

    else:

        for i, recommendation in enumerate(
            recs,
            start=1,
        ):

            st.markdown(
                f"""
                <div class="recommendation">

                    <span class="recommendation-number">
                        {i:02d}
                    </span>

                    &nbsp;&nbsp;

                    {recommendation}

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
            "Monitor AQI and applicable local environmental guidance."
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

            if isinstance(
                alert,
                dict,
            ):

                title = alert.get(
                    "title",
                    "Environmental Alert",
                )

                message = alert.get(
                    "message",
                    "",
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

        summary = alert_summary(
            aqi,
            risk_score,
            environment,
        )

        st.write(
            summary
        )

    except Exception:
        pass

    try:

        critical = has_critical_alerts(
            aqi,
            risk_score,
        )

        if critical:

            st.error(
                "🔴 Critical environmental conditions detected."
            )

    except Exception:
        pass


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
        "Timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "AQI": round(aqi, 2),
        "AQI Category": aqi_status,
        "Predicted AQI": round(
            predicted_aqi,
            2,
        ),
        "Risk Score": round(
            risk_score,
            2,
        ),
        "Risk Level": risk_status,
        "PM2.5": environment.get(
            "PM2.5",
            0,
        ),
        "PM10": environment.get(
            "PM10",
            0,
        ),
        "NO2": environment.get(
            "NO2",
            0,
        ),
        "SO2": environment.get(
            "SO2",
            0,
        ),
        "CO": environment.get(
            "CO",
            0,
        ),
        "O3": environment.get(
            "O3",
            0,
        ),
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

    filename = (
        "greenguardian_"
        + selected_city.lower().replace(" ", "_")
        + "_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".csv"
    )

    st.download_button(
        label="⬇️ Download Environmental Report",
        data=csv_data,
        file_name=filename,
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
