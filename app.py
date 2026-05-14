import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Seoul Bike Demand Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SeoulBikeData.csv"
MODEL_PATH = BASE_DIR / "models" / "bike_demand_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

PRIMARY = "#38BDF8"
ACCENT = "#A78BFA"
SUCCESS = "#22C55E"
WARNING = "#F97316"
TEXT = "#F8FAFC"
MUTED = "#CBD5E1"
BG = "#020617"

st.markdown(
    f"""
<style>
.stApp {{
    background: radial-gradient(circle at top left, #111827 0, {BG} 38%, #020617 100%);
    color: {TEXT};
}}
.block-container {{
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1480px;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0F172A 0%, #020617 100%);
    border-right: 1px solid #1E293B;
}}
h1, h2, h3, h4, label, p, span {{ color: {TEXT}; }}
.hero {{
    padding: 2.3rem 2.6rem;
    border-radius: 30px;
    background: linear-gradient(135deg, rgba(14,165,233,.26), rgba(79,70,229,.22), rgba(2,6,23,.95));
    border: 1px solid rgba(148,163,184,.24);
    box-shadow: 0 22px 70px rgba(2,6,23,.42);
    margin-bottom: 1.25rem;
}}
.hero-badge {{
    display:inline-block;
    color:#BAE6FD;
    background: rgba(14,165,233,.16);
    border: 1px solid rgba(56,189,248,.42);
    padding:.45rem .85rem;
    border-radius:999px;
    font-size:.82rem;
    font-weight:800;
    margin-bottom:1rem;
}}
.hero h1 {{
    font-size: 2.7rem;
    margin: 0 0 .7rem 0;
    letter-spacing: -.04em;
}}
.hero p {{
    color: {MUTED};
    max-width: 940px;
    font-size: 1.06rem;
    line-height: 1.7;
}}
.kpi-card {{
    background: linear-gradient(180deg, rgba(30,41,59,.94), rgba(15,23,42,.94));
    border: 1px solid rgba(148,163,184,.24);
    border-radius: 22px;
    padding: 1rem 1.1rem;
    min-height: 122px;
    box-shadow: 0 12px 32px rgba(2,6,23,.28);
}}
.kpi-label {{ color:#BAE6FD; font-size:.82rem; font-weight:800; margin-bottom:.4rem; }}
.kpi-value {{ color:white; font-size:2rem; font-weight:950; letter-spacing:-.04em; }}
.kpi-sub {{ color:#86EFAC; font-size:.83rem; margin-top:.3rem; }}
.insight {{
    background: rgba(15,23,42,.78);
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 18px;
    padding: 1rem;
    height: 100%;
}}
.insight strong {{color:white;}}
[data-testid="stMetric"] {{
    background: linear-gradient(180deg, rgba(30,41,59,.92), rgba(15,23,42,.92));
    border: 1px solid rgba(148,163,184,.22);
    padding: 16px;
    border-radius: 18px;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; border-bottom: 1px solid #1E293B; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 999px;
    padding: 10px 18px;
    background: #111827;
    border: 1px solid #334155;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(90deg, #0EA5E9, #4F46E5) !important;
    color: white !important;
}}
div[data-testid="stDataFrame"] {{ border: 1px solid #1E293B; border-radius: 16px; }}
.footer {{
    color: #94A3B8;
    text-align:center;
    padding: 2rem 0 0 0;
    font-size:.9rem;
}}
</style>
""",
    unsafe_allow_html=True,
)


def clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.replace("_", " ").title() for c in out.columns]
    return out


@st.cache_data
def load_raw_data() -> pd.DataFrame:
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
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
    df["is_holiday"] = (df["holiday"] == "Holiday").astype(int)
    df["day_type"] = np.where((df["is_weekend"] == 1) | (df["is_holiday"] == 1), "Leisure", "Work")
    return df


def prepare_features(raw_df: pd.DataFrame):
    features_df = raw_df.drop(columns=["rented_bike_count", "date", "weekday", "dew_point_temperature", "month_name"])
    features_df = pd.get_dummies(
        features_df,
        columns=["season", "holiday", "functioning_day", "day_type"],
        drop_first=True,
    )
    features_df = features_df.astype(float)
    target = raw_df["rented_bike_count"]
    return features_df, target


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


@st.cache_resource
def load_or_train_artifact(X: pd.DataFrame, y: pd.Series):
    if MODEL_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            if "model" in artifact and "feature_columns" in artifact and "metrics" in artifact:
                return artifact
        except Exception:
            pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=2),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    metrics = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = evaluate(y_test, pred)
        fitted[name] = model
    best_name = max(metrics, key=lambda name: metrics[name]["r2"])
    return {
        "model": fitted[best_name],
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "metrics": metrics,
        "feature_importance": dict(zip(X.columns, map(float, fitted[best_name].feature_importances_))) if hasattr(fitted[best_name], "feature_importances_") else None,
    }


def make_model_input(hour, temp, humidity, wind, visibility, solar, rainfall, snowfall, season, holiday, functioning_day, day_type, feature_columns):
    row = {c: 0 for c in feature_columns}
    row.update(
        {
            "hour": hour,
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind,
            "visibility": visibility,
            "solar_radiation": solar,
            "rainfall": rainfall,
            "snowfall": snowfall,
            "month": 6,
            "day": 15,
            "is_weekend": 1 if day_type == "Leisure" else 0,
            "is_holiday": 1 if holiday == "Holiday" else 0,
        }
    )
    for col in [f"season_{season}", f"holiday_{holiday}", f"functioning_day_{functioning_day}", f"day_type_{day_type}"]:
        if col in row:
            row[col] = 1
    return pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)


def plotly_layout(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=10, r=10, t=55, b=10),
        height=height,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.16)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.16)")
    return fig


def kpi(label, value, sub=""):
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


raw = load_raw_data()
X, y = prepare_features(raw)
artifact = load_or_train_artifact(X, y)
model = artifact["model"]
model_name = artifact.get("model_name", "Random Forest")
feature_columns = artifact["feature_columns"]
metrics_df = pd.DataFrame(
    [{"Model": name, **vals} for name, vals in artifact["metrics"].items()]
)
best_row = metrics_df.sort_values("r2", ascending=False).iloc[0]

st.sidebar.markdown("## 🚲 Seoul Bike AI")
st.sidebar.caption("Urban mobility demand intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("**Source**  ")
st.sidebar.caption("SeoulBikeData.csv uploaded with this project.")
st.sidebar.markdown("**Production model**  ")
st.sidebar.caption(f"{model_name} trained from the source dataset.")

st.markdown(
    """
<div class="hero">
  <div class="hero-badge">Portfolio ML Product · Seoul Urban Mobility</div>
  <h1>🚲 Seoul Bike Demand Intelligence</h1>
  <p>AI-powered forecasting dashboard for hourly bike rental demand, combining weather, seasonality, and operational context into business-ready mobility insights.</p>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Total Rentals", f"{int(raw['rented_bike_count'].sum()):,}", "from source dataset")
with c2:
    kpi("Best R²", f"{best_row['r2']:.3f}", best_row["Model"])
with c3:
    kpi("Records", f"{len(raw):,}", "hourly observations")
with c4:
    peak_hour = int(raw.groupby("hour")["rented_bike_count"].mean().idxmax())
    kpi("Peak Demand Hour", f"{peak_hour:02d}:00", "highest avg demand")

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Demand Predictor", "📈 Operational Insights", "🏆 Model Performance", "🧾 Data Explorer"])

with tab1:
    st.markdown("## 🔮 Predict hourly rental demand")
    st.caption(f"Adjust the operating scenario below. Prediction uses {model_name} trained from SeoulBikeData.csv.")
    left, mid, right = st.columns(3)
    with left:
        hour = st.slider("Hour of day", 0, 23, 8)
        temp = st.slider("Temperature (°C)", float(raw["temperature"].min()), float(raw["temperature"].max()), 25.0, 0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        wind = st.slider("Wind speed (m/s)", 0.0, 8.0, 1.5, 0.1)
    with mid:
        visibility = st.slider("Visibility (10m)", int(raw["visibility"].min()), int(raw["visibility"].max()), 1500, 10)
        solar = st.slider("Solar radiation (MJ/m²)", 0.0, float(raw["solar_radiation"].max()), 0.5, 0.01)
        rainfall = st.slider("Rainfall (mm)", 0.0, float(raw["rainfall"].max()), 0.0, 0.1)
        snowfall = st.slider("Snowfall (cm)", 0.0, float(raw["snowfall"].max()), 0.0, 0.1)
    with right:
        season = st.selectbox("Season", sorted(raw["season"].unique()), index=list(sorted(raw["season"].unique())).index("Summer"))
        holiday = st.selectbox("Holiday", sorted(raw["holiday"].unique()))
        functioning_day = st.selectbox("Functioning day", sorted(raw["functioning_day"].unique()), index=list(sorted(raw["functioning_day"].unique())).index("Yes"))
        day_type = st.selectbox("Day type", ["Work", "Leisure"])

    model_input = make_model_input(
        hour, temp, humidity, wind, visibility, solar, rainfall, snowfall, season, holiday, functioning_day, day_type, feature_columns
    )
    prediction = max(0, float(model.predict(model_input)[0]))
    q25, q75 = raw["rented_bike_count"].quantile([0.25, 0.75])
    demand_label = "High" if prediction >= q75 else ("Low" if prediction <= q25 else "Moderate")
    gauge_color = SUCCESS if demand_label == "High" else (WARNING if demand_label == "Moderate" else PRIMARY)

    a, b = st.columns([1.1, 1])
    with a:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={"suffix": " bikes", "font": {"size": 42}},
                title={"text": f"Expected Demand · {demand_label}"},
                gauge={
                    "axis": {"range": [0, max(3500, raw["rented_bike_count"].max())]},
                    "bar": {"color": gauge_color},
                    "bgcolor": "rgba(15,23,42,.85)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(148,163,184,.25)",
                    "steps": [
                        {"range": [0, q25], "color": "rgba(56,189,248,.18)"},
                        {"range": [q25, q75], "color": "rgba(249,115,22,.20)"},
                        {"range": [q75, raw["rented_bike_count"].max()], "color": "rgba(34,197,94,.20)"},
                    ],
                },
            )
        )
        st.plotly_chart(plotly_layout(fig, 330), use_container_width=True)
    with b:
        st.markdown("### Recommended action")
        if demand_label == "High":
            msg = "Prepare additional bikes and rebalance before peak time. Increase station monitoring and maintenance readiness."
        elif demand_label == "Low":
            msg = "Demand is expected to be light. This is a good window for maintenance, inspections, and inventory balancing."
        else:
            msg = "Standard operations should be enough. Monitor nearby peak periods and weather changes."
        st.markdown(f"<div class='insight'><strong>{demand_label} demand scenario</strong><br><br>{msg}</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("## 📈 Operational insights")
    st.caption("All charts below are generated directly from the uploaded source CSV.")
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.metric("Average Demand", f"{raw['rented_bike_count'].mean():,.0f}")
    with i2:
        st.metric("Max Demand", f"{raw['rented_bike_count'].max():,}")
    with i3:
        st.metric("Highest Season", raw.groupby("season")["rented_bike_count"].mean().idxmax())
    with i4:
        st.metric("Functioning Days", f"{(raw['functioning_day'] == 'Yes').mean()*100:.1f}%")

    hourly = raw.groupby("hour", as_index=False)["rented_bike_count"].mean()
    fig_hour = px.line(hourly, x="hour", y="rented_bike_count", markers=True, title="Average demand by hour")
    fig_hour.update_traces(line=dict(color=PRIMARY, width=3), marker=dict(size=8))
    st.plotly_chart(plotly_layout(fig_hour, 390), use_container_width=True)

    col1, col2 = st.columns(2)
    season_avg = raw.groupby("season", as_index=False)["rented_bike_count"].mean().sort_values("rented_bike_count", ascending=False)
    fig_season = px.bar(season_avg, x="season", y="rented_bike_count", title="Average demand by season", color="rented_bike_count", color_continuous_scale="Blues")
    col1.plotly_chart(plotly_layout(fig_season, 390), use_container_width=True)
    sample_scatter = raw.sample(min(2500, len(raw)), random_state=42)
    fig_temp = px.scatter(
        sample_scatter,
        x="temperature",
        y="rented_bike_count",
        color="season",
        title="Temperature vs. rentals",
        opacity=0.72,
        color_discrete_sequence=[PRIMARY, ACCENT, "#F472B6", "#FB7185"],
    )
    col2.plotly_chart(plotly_layout(fig_temp, 390), use_container_width=True)

    st.markdown("### Business interpretation")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("<div class='insight'><strong>Peak-hour planning</strong><br><br>Use the hourly trend to schedule rebalancing before demand spikes.</div>", unsafe_allow_html=True)
    with b2:
        st.markdown("<div class='insight'><strong>Weather sensitivity</strong><br><br>Temperature and seasonality are strong demand drivers and should be monitored daily.</div>", unsafe_allow_html=True)
    with b3:
        st.markdown("<div class='insight'><strong>Operational availability</strong><br><br>Functioning-day status has a large impact and should be part of demand planning.</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("## 🏆 Model performance")
    st.caption("Metrics are calculated from the uploaded SeoulBikeData.csv using the project preprocessing and train/test split logic.")
    display = metrics_df.sort_values("r2", ascending=False).copy()
    st.dataframe(
        display.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R2": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )
    m1, m2 = st.columns(2)
    fig_r2 = px.bar(display, x="Model", y="R2", text="R2", color="R2", title="R² score by model", color_continuous_scale="Blues")
    fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    m1.plotly_chart(plotly_layout(fig_r2, 390), use_container_width=True)
    fig_mae = px.bar(display.sort_values("MAE"), x="Model", y="MAE", text="MAE", color="MAE", title="MAE by model", color_continuous_scale="Purples_r")
    fig_mae.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    m2.plotly_chart(plotly_layout(fig_mae, 390), use_container_width=True)

    importance_data = artifact.get("feature_importance")
    if importance_data:
        importance = pd.DataFrame({"Feature": list(importance_data.keys()), "Importance": list(importance_data.values())}).sort_values("Importance", ascending=False).head(10)
        fig_imp = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top 10 feature importances ({model_name})",
            color="Importance",
            color_continuous_scale="Teal",
        )
        st.plotly_chart(plotly_layout(fig_imp, 520), use_container_width=True)

with tab4:
    st.markdown("## 🧾 Data explorer")
    st.caption("Filter the uploaded Seoul Bike dataset and download the filtered result.")
    f1, f2, f3 = st.columns(3)
    with f1:
        seasons = st.multiselect("Season", sorted(raw["season"].unique()), default=sorted(raw["season"].unique()))
    with f2:
        hours = st.slider("Hour range", 0, 23, (0, 23))
    with f3:
        temp_range = st.slider(
            "Temperature range",
            float(raw["temperature"].min()),
            float(raw["temperature"].max()),
            (float(raw["temperature"].min()), float(raw["temperature"].max())),
        )
    filtered = raw[
        raw["season"].isin(seasons)
        & raw["hour"].between(hours[0], hours[1])
        & raw["temperature"].between(temp_range[0], temp_range[1])
    ]
    st.metric("Filtered Records", f"{len(filtered):,}")
    st.dataframe(clean_dataframe_for_display(filtered.head(200)), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_seoul_bike_data.csv",
        mime="text/csv",
    )

st.markdown("<div class='footer'>Built with Streamlit · Scikit-learn · Plotly · Seoul Bike Sharing Demand Dataset</div>", unsafe_allow_html=True)
