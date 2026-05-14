import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SeoulBikeData.csv"
MODEL_PATH = ROOT / "models" / "bike_demand_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"


def load_and_prepare_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(path, encoding="unicode_escape")
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
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
    df["is_holiday"] = (df["holiday"] == "Holiday").astype(int)
    df["day_type"] = np.where((df["is_weekend"] == 1) | (df["is_holiday"] == 1), "Leisure", "Work")

    raw_df = df.copy()
    features_df = df.drop(columns=["rented_bike_count", "date", "weekday", "dew_point_temperature"])
    features_df = pd.get_dummies(features_df, columns=["season", "holiday", "functioning_day", "day_type"], drop_first=True)
    features_df = features_df.astype(float)
    target = df["rented_bike_count"]
    return features_df, target, raw_df


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    X, y, raw_df = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=2),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    metrics = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = evaluate(y_test, pred)
        fitted_models[name] = model

    best_name = max(metrics, key=lambda name: metrics[name]["r2"])
    best_model = fitted_models[best_name]

    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(X.columns, map(float, best_model.feature_importances_)))

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "metrics": metrics,
        "feature_importance": feature_importance,
    }
    joblib.dump(artifact, MODEL_PATH)

    summary = {
        "best_model": best_name,
        "metrics": metrics,
        "rows": int(raw_df.shape[0]),
        "features": int(X.shape[1]),
        "target_mean": float(y.mean()),
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
