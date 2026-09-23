import os
import sys
import pandas as pd
import streamlit as st

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# =========================================================
# PROJECT IMPORTS
# =========================================================

from src.data_loader import (
    load_pollution_data,
    load_weather_data,
    load_city_data,
    get_city_data,
)

from src.pollution_api import (
    build_environment_record,
    air_quality_history,
)

from src.aqi_calculator import (
    calculate_aqi,
    aqi_description,
    aqi_color,
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

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI GreenGuardian",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #07130f;
        color: #f1f5f3;
    }

    section[data-testid="stSidebar"] {
        background: #0b1f18;
        border-right: 1px solid #18382c;
    }

    section[data-testid="stSidebar"] * {
        color: #e7f5ee !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #e8fff2 !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #0d241b,
            #102c21
        );
        border: 1px solid #1b4938;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.20);
    }

    div[data-testid="stMetricLabel"] {
        color: #9fc4b2 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #eafff2 !important;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #245640;
        background: #123a2a;
        color: white;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #49d18c;
        color: white;
    }

    div[data-baseweb="select"] > div {
        background: #0d241b;
        border-color: #245640;
    }

    .info-card {
        background: #0d241b;
        border: 1px solid #1d4636;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .status-online {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        background: #123d2b;
        color: #72e6a9;
        font-size: 13px;
        font-weight: 600;
    }

    .status-demo {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        background: #3b2f12;
        color: #ffd76a;
        font-size: 13px;
        font-weight: 600;
    }

    .section-divider {
        height: 1px;
        background: #1a3b2f;
        margin: 25px 0;
    }

    .footer {
        text-align: center;
        color: #729586;
        padding: 30px 0 10px 0;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CITY COORDINATES
# =========================================================

CITY_COORDINATES = {
    "Lucknow": (26.8467, 80.9462),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kanpur": (26.4499, 80.3319),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def get_aqi_category(aqi):

    aqi = safe_float(aqi)

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


def safe_call(function, *args, default=None, **kwargs):

    try:
        return function(*args, **kwargs)

    except Exception:
        return default


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_all_data():

    try:
        pollution = load_pollution_data()
    except Exception:
        pollution = pd.DataFrame()

    try:
        weather = load_weather_data()
    except Exception:
        weather = pd.DataFrame()

    try:
        cities = load_city_data()
    except Exception:
        cities = pd.DataFrame()

    return pollution, weather, cities


pollution_df, weather_df, city_df = load_all_data()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:10px 0 20px 0;
        ">
            <div style="font-size:42px;">🌱</div>

            <h2 style="margin:0;">
                GreenGuardian
            </h2>

            <p style="
                color:#82ad9b;
                font-size:13px;
                margin-top:5px;
            ">
                Environmental Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

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

    st.markdown("---")

    # =====================================================
    # CITY LIST
    # =====================================================

    default_cities = list(CITY_COORDINATES.keys())

    available_cities = default_cities.copy()

    try:

        if city_df is not None and not city_df.empty:

            for column in [
                "City",
                "city",
                "CITY",
                "Name",
                "name",
            ]:

                if column in city_df.columns:

                    values = (
                        city_df[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    available_cities = sorted(
                        list(
                            set(
                                default_cities
                                + values
                            )
                        )
                    )

                    break

    except Exception:
        pass

    selected_city = st.selectbox(
        "📍 Select City",
        available_cities,
        index=(
            available_cities.index("Lucknow")
            if "Lucknow" in available_cities
            else 0
        ),
    )

    live_mode = st.toggle(
        "🌐 Use Live Data",
        value=False,
    )

    if st.button(
        "🔄 Refresh Data",
        use_container_width=True,
    ):

        st.cache_data.clear()
        st.rerun()

# =========================================================
# ENVIRONMENT DATA
# =========================================================

environment = {}
data_source = "Dataset Demo"

# =========================================================
# LIVE OPEN-METEO DATA
# =========================================================

if live_mode:

    try:

        latitude, longitude = CITY_COORDINATES.get(
            selected_city,
            (26.8467, 80.9462),
        )

        # IMPORTANT:
        # build_environment_record requires:
        # city, latitude, longitude

        environment = build_environment_record(
            selected_city,
            latitude,
            longitude,
        )

        if environment is None:
            raise ValueError(
                "Live API returned empty data."
            )

        data_source = "Live Open-Meteo"

    except Exception as error:

        st.warning(
            f"Live data unavailable. "
            f"Using dataset data instead. "
            f"({error})"
        )

        live_mode = False
        environment = {}
        data_source = "Dataset Demo"

# =========================================================
# DATASET FALLBACK
# =========================================================

if not environment:

    try:

        city_info = get_city_data(
            city_df,
            selected_city,
        )

        if isinstance(
            city_info,
            pd.DataFrame,
        ):

            if not city_info.empty:
                city_info = city_info.iloc[0]

        if isinstance(
            city_info,
            pd.Series,
        ):

            environment = city_info.to_dict()

    except Exception:
        environment = {}

# =========================================================
# POLLUTION DATASET FALLBACK
# =========================================================

if not environment and not pollution_df.empty:

    try:

        city_column = None

        for column in [
            "City",
            "city",
            "CITY",
            "Name",
            "name",
        ]:

            if column in pollution_df.columns:
                city_column = column
                break

        if city_column:

            filtered = pollution_df[
                pollution_df[city_column]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                selected_city
                .strip()
                .lower()
            ]

            if not filtered.empty:

                environment = (
                    filtered.iloc[-1]
                    .to_dict()
                )

    except Exception:
        pass

# =========================================================
# DEFAULT VALUES
# =========================================================

environment.setdefault("PM2.5", 75)
environment.setdefault("PM10", 120)
environment.setdefault("NO2", 40)
environment.setdefault("SO2", 20)
environment.setdefault("CO", 0.8)
environment.setdefault("O3", 50)

environment.setdefault(
    "Temperature",
    28,
)

environment.setdefault(
    "Humidity",
    60,
)

environment.setdefault(
    "Wind Speed",
    8,
)

# =========================================================
# CLEAN DATA
# =========================================================

try:

    environment = clean_environment_data(
        environment
    )

except Exception:
    pass

# =========================================================
# POLLUTANT VALUES
# =========================================================

pm25 = safe_float(
    environment.get("PM2.5")
)

pm10 = safe_float(
    environment.get("PM10")
)

no2 = safe_float(
    environment.get("NO2")
)

so2 = safe_float(
    environment.get("SO2")
)

co = safe_float(
    environment.get("CO")
)

o3 = safe_float(
    environment.get("O3")
)

temperature = safe_float(
    environment.get("Temperature")
)

humidity = safe_float(
    environment.get("Humidity")
)

wind_speed = safe_float(
    environment.get("Wind Speed")
)

# =========================================================
# AQI
# =========================================================

try:

    aqi = calculate_aqi(
        pm25=pm25,
        pm10=pm10,
        no2=no2,
        so2=so2,
        co=co,
        o3=o3,
    )

except Exception:

    aqi = max(
        pm25,
        pm10 * 0.65,
        no2 * 1.5,
        so2 * 2,
        o3 * 1.2,
        co * 50,
    )

aqi = round(
    safe_float(aqi),
    1,
)

try:

    aqi_desc = aqi_description(aqi)

except Exception:

    aqi_desc = get_aqi_category(aqi)

try:

    aqi_col = aqi_color(aqi)

except Exception:

    aqi_col = "#49d18c"

aqi_category = get_aqi_category(aqi)

# =========================================================
# ENVIRONMENTAL RISK
# =========================================================

try:

    risk_score = environmental_risk(
        environment
    )

except Exception:

    risk_score = min(
        100,
        (
            pm25 * 0.35
            + pm10 * 0.20
            + no2 * 0.15
            + so2 * 0.10
            + o3 * 0.10
            + co * 10
        ),
    )

risk_score = round(
    safe_float(risk_score),
    1,
)

try:

    risk = risk_level(
        risk_score
    )

except Exception:

    if risk_score < 25:
        risk = "Low"

    elif risk_score < 50:
        risk = "Moderate"

    elif risk_score < 75:
        risk = "High"

    else:
        risk = "Critical"

try:

    risk_msg = risk_message(
        risk_score
    )

except Exception:

    risk_msg = (
        "Monitor environmental conditions "
        "and follow recommended precautions."
    )

# =========================================================
# MAIN HEADER
# =========================================================

header_col1, header_col2 = st.columns(
    [7, 3]
)

with header_col1:

    st.title(
        "🌱 AI GreenGuardian"
    )

    st.caption(
        "Smart Pollution Monitoring • "
        "AI Prediction • "
        "Environmental Risk Intelligence"
    )

with header_col2:

    if live_mode:

        st.markdown(
            '<div class="status-online">'
            '● LIVE DATA'
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="status-demo">'
            '● DATASET MODE'
            '</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"📍 {selected_city}"
    )

# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.header(
        "Environmental Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "AQI",
            f"{aqi:.0f}",
            aqi_category,
        )

    with col2:

        st.metric(
            "PM2.5",
            f"{pm25:.1f} µg/m³",
        )

    # =====================================================
    # AI PREDICTION
    # =====================================================

    try:

        predicted_aqi = predict_next_aqi(
            environment,
            selected_city,
        )

    except Exception:

        predicted_aqi = aqi * 1.03

    predicted_aqi = safe_float(
        predicted_aqi,
        aqi,
    )

    prediction_delta = (
        predicted_aqi - aqi
    )

    with col3:

        st.metric(
            "Predicted AQI",
            f"{predicted_aqi:.0f}",
            f"{prediction_delta:+.0f}",
        )

    with col4:

        st.metric(
            "Risk Score",
            f"{risk_score:.0f}/100",
            risk,
        )

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    # =====================================================
    # GAUGES
    # =====================================================

    gauge1, gauge2 = st.columns(2)

    with gauge1:

        st.subheader(
            "AQI Status"
        )

        try:

            fig = aqi_gauge(aqi)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        except Exception:

            st.info(
                f"AQI: {aqi:.0f} — "
                f"{aqi_category}"
            )

    with gauge2:

        st.subheader(
            "Environmental Risk"
        )

        try:

            fig = risk_gauge(
                risk_score
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        except Exception:

            st.progress(
                min(
                    risk_score / 100,
                    1.0,
                )
            )

            st.write(
                f"Risk: {risk}"
            )

    # =====================================================
    # POLLUTION CHART
    # =====================================================

    st.subheader(
        "Pollution Levels"
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
                pm25,
                pm10,
                no2,
                so2,
                co,
                o3,
            ],
        }
    )

    try:

        fig = pollutant_bar_chart(
            pollutant_data
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    except Exception:

        st.bar_chart(
            pollutant_data.set_index(
                "Pollutant"
            )
        )

    # =====================================================
    # WEATHER
    # =====================================================

    st.subheader(
        "Weather Conditions"
    )

    weather_values = pd.DataFrame(
        {
            "Metric": [
                "Temperature",
                "Humidity",
                "Wind Speed",
            ],
            "Value": [
                temperature,
                humidity,
                wind_speed,
            ],
        }
    )

    st.dataframe(
        weather_values,
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # RISK MESSAGE
    # =====================================================

    st.markdown(
        f"""
        <div class="info-card">
            <h4>⚠️ Environmental Risk Status</h4>
            <p>{risk_msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# POLLUTION MONITOR
# =========================================================

elif page == "🌫 Pollution Monitor":

    st.title(
        "🌫 Pollution Monitor"
    )

    st.write(
        f"Current pollution profile for **{selected_city}**."
    )

    values = [
        ("PM2.5", pm25, "µg/m³"),
        ("PM10", pm10, "µg/m³"),
        ("NO₂", no2, "µg/m³"),
        ("SO₂", so2, "µg/m³"),
        ("CO", co, "mg/m³"),
        ("O₃", o3, "µg/m³"),
    ]

    cols = st.columns(6)

    for col, item in zip(
        cols,
        values,
    ):

        name, value, unit = item

        with col:

            st.metric(
                name,
                f"{value:.1f}",
                unit,
            )

    st.markdown("---")

    st.subheader(
        "Pollutant Distribution"
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
                pm25,
                pm10,
                no2,
                so2,
                co,
                o3,
            ],
        }
    )

    try:

        fig = pollutant_bar_chart(
            pollutant_data
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    except Exception:

        st.bar_chart(
            pollutant_data.set_index(
                "Pollutant"
            )
        )

    st.subheader(
        "Pollution History"
    )

    try:

        history = air_quality_history(
            selected_city
        )

        if (
            history is not None
            and not history.empty
        ):

            try:

                fig = pollution_chart(
                    history
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            except Exception:

                st.dataframe(
                    history,
                    use_container_width=True,
                )

        else:

            st.info(
                "Historical pollution data "
                "is not available."
            )

    except Exception:

        st.info(
            "Historical pollution data "
            "is not available."
        )

# =========================================================
# AI PREDICTION
# =========================================================

elif page == "🤖 AI Prediction":

    st.title(
        "🤖 AI Pollution Prediction"
    )

    st.write(
        "AI-based estimation of future environmental conditions."
    )

    try:

        predicted_aqi = predict_next_aqi(
            environment,
            selected_city,
        )

    except Exception:

        predicted_aqi = aqi * 1.03

    predicted_aqi = safe_float(
        predicted_aqi,
        aqi,
    )

    change = (
        predicted_aqi - aqi
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Current AQI",
            f"{aqi:.0f}",
        )

    with c2:

        st.metric(
            "Predicted AQI",
            f"{predicted_aqi:.0f}",
            f"{change:+.0f}",
        )

    with c3:

        if change > 1:
            trend = "Increasing"

        elif change < -1:
            trend = "Decreasing"

        else:
            trend = "Stable"

        st.metric(
            "Trend",
            trend,
        )

    st.markdown("---")

    prediction_df = pd.DataFrame(
        {
            "Metric": [
                "Current AQI",
                "Predicted AQI",
                "AQI Change",
            ],
            "Value": [
                aqi,
                predicted_aqi,
                change,
            ],
        }
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f"""
        <div class="info-card">
            <h4>🤖 AI Prediction</h4>

            <p>
                Current AQI:
                <strong>{aqi:.0f}</strong>
            </p>

            <p>
                Estimated AQI:
                <strong>{predicted_aqi:.0f}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# RISK INTELLIGENCE
# =========================================================

elif page == "⚠️ Risk Intelligence":

    st.title(
        "⚠️ Environmental Risk Intelligence"
    )

    st.write(
        "AI-assisted environmental risk assessment."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Risk Score",
            f"{risk_score:.0f}/100",
        )

        st.metric(
            "Risk Level",
            risk,
        )

    with c2:

        try:

            fig = risk_gauge(
                risk_score
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        except Exception:

            st.progress(
                min(
                    risk_score / 100,
                    1.0,
                )
            )

    st.markdown("---")

    st.subheader(
        "Risk Explanation"
    )

    st.markdown(
        f"""
        <div class="info-card">
            <h4>Current Assessment</h4>
            <p>{risk_msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk_table = pd.DataFrame(
        {
            "Indicator": [
                "PM2.5",
                "PM10",
                "NO2",
                "SO2",
                "CO",
                "O3",
                "Temperature",
                "Humidity",
                "Wind Speed",
            ],
            "Value": [
                pm25,
                pm10,
                no2,
                so2,
                co,
                o3,
                temperature,
                humidity,
                wind_speed,
            ],
        }
    )

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# CITY COMPARISON
# =========================================================

elif page == "🏙 City Comparison":

    st.title(
        "🏙 City Comparison"
    )

    st.write(
        "Compare pollution levels across cities."
    )

    comparison_cities = st.multiselect(
        "Select Cities",
        available_cities,
        default=available_cities[:5],
    )

    if comparison_cities:

        rows = []

        for city in comparison_cities:

            city_environment = {}

            try:

                info = get_city_data(
                    city_df,
                    city,
                )

                if isinstance(
                    info,
                    pd.DataFrame,
                ):

                    if not info.empty:
                        info = info.iloc[0]

                if isinstance(
                    info,
                    pd.Series,
                ):

                    city_environment = (
                        info.to_dict()
                    )

            except Exception:
                pass

            city_environment.setdefault(
                "PM2.5",
                75,
            )

            city_environment.setdefault(
                "PM10",
                120,
            )

            city_environment.setdefault(
                "NO2",
                40,
            )

            city_environment.setdefault(
                "SO2",
                20,
            )

            city_environment.setdefault(
                "CO",
                0.8,
            )

            city_environment.setdefault(
                "O3",
                50,
            )

            try:

                city_aqi = calculate_aqi(
                    pm25=safe_float(
                        city_environment[
                            "PM2.5"
                        ]
                    ),
                    pm10=safe_float(
                        city_environment[
                            "PM10"
                        ]
                    ),
                    no2=safe_float(
                        city_environment[
                            "NO2"
                        ]
                    ),
                    so2=safe_float(
                        city_environment[
                            "SO2"
                        ]
                    ),
                    co=safe_float(
                        city_environment[
                            "CO"
                        ]
                    ),
                    o3=safe_float(
                        city_environment[
                            "O3"
                        ]
                    ),
                )

            except Exception:

                city_aqi = safe_float(
                    city_environment[
                        "PM2.5"
                    ]
                )

            rows.append(
                {
                    "City": city,
                    "AQI": round(
                        safe_float(
                            city_aqi
                        ),
                        1,
                    ),
                    "PM2.5": safe_float(
                        city_environment[
                            "PM2.5"
                        ]
                    ),
                    "PM10": safe_float(
                        city_environment[
                            "PM10"
                        ]
                    ),
                    "NO2": safe_float(
                        city_environment[
                            "NO2"
                        ]
                    ),
                }
            )

        comparison_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        try:

            fig = city_comparison_chart(
                comparison_df
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        except Exception:

            st.bar_chart(
                comparison_df.set_index(
                    "City"
                )["AQI"]
            )

        st.subheader(
            "Pollutant Comparison"
        )

        try:

            fig = pollutant_comparison_chart(
                comparison_df
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        except Exception:
            pass

    else:

        st.info(
            "Select at least one city."
        )

# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "🧠 Explainable AI":

    st.title(
        "🧠 Explainable AI"
    )

    st.write(
        "Understand which environmental indicators contribute to the assessment."
    )

    try:

        contribution_df = (
            contribution_dataframe(
                environment
            )
        )

        if (
            contribution_df is not None
            and not contribution_df.empty
        ):

            st.subheader(
                "Feature Contributions"
            )

            st.dataframe(
                contribution_df,
                use_container_width=True,
                hide_index=True,
            )

            try:

                fig = contribution_chart(
                    contribution_df
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            except Exception:
                pass

            try:

                contributors = (
                    top_contributors(
                        contribution_df
                    )
                )

                if contributors:

                    st.subheader(
                        "Top Contributors"
                    )

                    for item in contributors:

                        st.write(
                            f"• {item}"
                        )

            except Exception:
                pass

        else:

            st.info(
                "Explainability information "
                "is not available."
            )

    except Exception:

        st.info(
            "Explainable AI model integration "
            "is currently unavailable."
        )

# =========================================================
# RECOMMENDATIONS
# =========================================================

elif page == "💡 Recommendations":

    st.title(
        "💡 Environmental Recommendations"
    )

    st.write(
        f"Recommendations for **{selected_city}**."
    )

    try:

        recs = recommendations(
            environment,
            aqi,
            risk_score,
        )

        if recs:

            for index, rec in enumerate(
                recs,
                start=1,
            ):

                st.markdown(
                    f"""
                    <div class="info-card">
                        <h4>
                            💡 Recommendation {index}
                        </h4>
                        <p>{rec}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No special recommendations."
            )

    except Exception:

        fallback = [
            "Monitor AQI before outdoor activities.",
            "Reduce unnecessary outdoor exposure during high pollution.",
            "Maintain good indoor ventilation.",
            "Consider appropriate respiratory protection during elevated pollution.",
        ]

        for rec in fallback:

            st.markdown(
                f"""
                <div class="info-card">
                    💡 {rec}
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# ALERTS
# =========================================================

elif page == "🚨 Alerts":

    st.title(
        "🚨 Environmental Alerts"
    )

    st.write(
        "Environmental warnings based on current conditions."
    )

    try:

        alerts = get_all_alerts(
            environment,
            aqi,
            risk_score,
        )

        if alerts:

            for alert in alerts:

                st.warning(
                    str(alert)
                )

        else:

            st.success(
                "✅ No major environmental alerts detected."
            )

    except Exception:

        if aqi >= 300:

            st.error(
                "🚨 Severe AQI alert detected."
            )

        elif aqi >= 200:

            st.warning(
                "⚠️ High pollution level detected."
            )

        elif aqi >= 100:

            st.info(
                "ℹ️ Moderate pollution level detected."
            )

        else:

            st.success(
                "✅ No major AQI alert."
            )

    try:

        critical = has_critical_alerts(
            environment,
            aqi,
            risk_score,
        )

        if critical:

            st.error(
                "🚨 Critical environmental condition detected."
            )

    except Exception:
        pass

# =========================================================
# DATA EXPORT
# =========================================================

elif page == "📥 Data Export":

    st.title(
        "📥 Data Export"
    )

    st.write(
        "Download the current environmental report."
    )

    export_df = pd.DataFrame(
        {
            "City": [selected_city],

            "Timestamp": [
                current_timestamp()
            ],

            "AQI": [aqi],

            "AQI Category": [
                aqi_category
            ],

            "PM2.5": [pm25],

            "PM10": [pm10],

            "NO2": [no2],

            "SO2": [so2],

            "CO": [co],

            "O3": [o3],

            "Temperature": [
                temperature
            ],

            "Humidity": [
                humidity
            ],

            "Wind Speed": [
                wind_speed
            ],

            "Risk Score": [
                risk_score
            ],

            "Risk Level": [
                risk
            ],

            "Data Source": [
                data_source
            ],
        }
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
            "green_guardian_"
            + selected_city.lower()
            .replace(" ", "_")
            + ".csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        🌱 <strong>AI GreenGuardian</strong><br>

        Smart Pollution Monitoring &
        Environmental Risk Intelligence<br>

        Python • Streamlit • Machine Learning • Open-Meteo

    </div>
    """,
    unsafe_allow_html=True,
)
