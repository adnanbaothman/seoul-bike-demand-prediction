from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SeoulBikeData.csv"
MODEL_PATH = ROOT / "models" / "bike_demand_model.pkl"

st.set_page_config(
    page_title="Seoul Bike Demand Intelligence",
    page_icon="🚲",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2rem;}
.metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 1.2rem;
    border-radius: 1rem;
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
}
.small-muted {color: #64748b; font-size: 0.9rem;}
.hero {
    padding: 1.6rem;
    border-radius: 1.25rem;
    background: linear-gradient(135deg, #ecfeff 0%, #f8fafc 55%, #f0fdf4 100%);
    border: 1px solid #e2e8f0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="unicode_escape")
    df = df.rename(
        columns={
            "Rented Bike Count": "rented_bike_count",
            "Temperature(°C)": "temperature",
            "Humidity(%)": "humidity",
            "Wind speed (m/s)": "wind_speed",
            "Visibility (10m)": "visibility",
            "Dew point temperature(°C)": "dew_point_temperature",
            "Solar Radiation (MJ/m2)": "solar_radiation",
            "Rainfall(mm)": "rainfall",
            "Snowfall (cm)": "snowfall",
            "Seasons": "season",
            "Holiday": "holiday",
            "Functioning Day": "functioning_day",
            "Hour": "hour",
            "Date": "date",
        }
    )
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["month_name"] = df["date"].dt.month_name()
    df["weekday"] = df["date"].dt.day_name()
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_input_row(feature_columns, values: dict) -> pd.DataFrame:
    row = {col: 0.0 for col in feature_columns}
    direct_values = {
        "hour": values["hour"],
        "temperature": values["temperature"],
        "humidity": values["humidity"],
        "wind_speed": values["wind_speed"],
        "visibility": values["visibility"],
        "solar_radiation": values["solar_radiation"],
        "rainfall": values["rainfall"],
        "snowfall": values["snowfall"],
        "month": values["month"],
        "day": values["day"],
        "is_weekend": 1.0 if values["day_type"] == "Leisure" else 0.0,
        "is_holiday": 1.0 if values["holiday"] == "Holiday" else 0.0,
    }
    for key, value in direct_values.items():
        if key in row:
            row[key] = float(value)

    encoded_flags = [
        f"season_{values['season']}",
        f"holiday_{values['holiday']}",
        f"functioning_day_{values['functioning_day']}",
        f"day_type_{values['day_type']}",
    ]
    for flag in encoded_flags:
        if flag in row:
            row[flag] = 1.0
    return pd.DataFrame([row], columns=feature_columns)


def demand_label(prediction: float) -> tuple[str, str]:
    if prediction >= 900:
        return "High demand", "Prepare extra bikes and staff coverage."
    if prediction >= 400:
        return "Medium demand", "Monitor stations and rebalance where needed."
    return "Low demand", "Standard operations should be enough."


df = load_data()
artifact = load_model()
model = artifact["model"]
model_name = artifact["model_name"]
feature_columns = artifact["feature_columns"]
metrics = artifact["metrics"]

st.markdown(
    """
    <div class="hero">
      <h1>🚲 Seoul Bike Demand Intelligence</h1>
      <p class="small-muted">A machine learning decision-support dashboard for predicting hourly bike rental demand using weather, time, and operations data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section", ["Demand Predictor", "City Insights", "Model Performance", "Data Explorer"])

if page == "Demand Predictor":
    st.subheader("Predict hourly rental demand")
    left, right = st.columns([1, 1])

    with left:
        hour = st.slider("Hour of day", 0, 23, 8)
        temperature = st.slider("Temperature (°C)", -20.0, 40.0, 25.0, 0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        wind_speed = st.slider("Wind speed (m/s)", 0.0, 8.0, 1.5, 0.1)
        visibility = st.slider("Visibility (10m)", 0, 2000, 1500)

    with right:
        solar_radiation = st.slider("Solar radiation (MJ/m²)", 0.0, 4.0, 0.5, 0.1)
        rainfall = st.slider("Rainfall (mm)", 0.0, 40.0, 0.0, 0.1)
        snowfall = st.slider("Snowfall (cm)", 0.0, 10.0, 0.0, 0.1)
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"], index=1)
        holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])
        functioning_day = st.selectbox("Functioning day", ["Yes", "No"])
        day_type = st.selectbox("Day type", ["Work", "Leisure"])
        month = st.slider("Month", 1, 12, 6)
        day = st.slider("Day", 1, 31, 15)

    input_df = build_input_row(
        feature_columns,
        {
            "hour": hour,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "visibility": visibility,
            "solar_radiation": solar_radiation,
            "rainfall": rainfall,
            "snowfall": snowfall,
            "season": season,
            "holiday": holiday,
            "functioning_day": functioning_day,
            "day_type": day_type,
            "month": month,
            "day": day,
        },
    )
    prediction = max(0, float(model.predict(input_df)[0]))
    label, recommendation = demand_label(prediction)

    metric_col, rec_col = st.columns([1, 2])
    with metric_col:
        st.metric("Predicted rentals", f"{prediction:,.0f} bikes/hour")
    with rec_col:
        st.info(f"**{label}:** {recommendation}")

elif page == "City Insights":
    st.subheader("Operational insights")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Average demand", f"{df['rented_bike_count'].mean():.0f}")
    c3.metric("Peak hour", int(df.groupby("hour")["rented_bike_count"].mean().idxmax()))
    c4.metric("Highest season", df.groupby("season")["rented_bike_count"].mean().idxmax())

    hourly = df.groupby("hour", as_index=False)["rented_bike_count"].mean()
    st.plotly_chart(px.line(hourly, x="hour", y="rented_bike_count", markers=True, title="Average demand by hour"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        season_df = df.groupby("season", as_index=False)["rented_bike_count"].mean().sort_values("rented_bike_count", ascending=False)
        st.plotly_chart(px.bar(season_df, x="season", y="rented_bike_count", title="Average demand by season"), use_container_width=True)
    with col2:
        st.plotly_chart(px.scatter(df, x="temperature", y="rented_bike_count", color="season", title="Temperature vs. rentals"), use_container_width=True)

elif page == "Model Performance":
    st.subheader("Model comparison")
    metric_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    st.dataframe(metric_df, use_container_width=True, hide_index=True)
    st.success(f"Selected production model: **{model_name}**")

    st.plotly_chart(px.bar(metric_df, x="model", y="r2", title="R² score by model", text_auto='.3f'), use_container_width=True)

    if artifact.get("feature_importance"):
        fi = pd.DataFrame(
            artifact["feature_importance"].items(), columns=["feature", "importance"]
        ).sort_values("importance", ascending=False).head(12)
        st.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h", title="Top feature importance"), use_container_width=True)

else:
    st.subheader("Data explorer")
    seasons = st.multiselect("Season", sorted(df["season"].unique()), default=sorted(df["season"].unique()))
    filtered = df[df["season"].isin(seasons)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download filtered data", filtered.to_csv(index=False), "filtered_bike_data.csv", "text/csv")
