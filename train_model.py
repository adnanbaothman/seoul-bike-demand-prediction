from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SeoulBikeData.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "bike_demand_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def prepare_data(path=DATA_PATH):
    df = pd.read_csv(path, encoding="unicode_escape")
    df = df.rename(columns={
        "Rented Bike Count": "RentedBike",
        "Temperature(°C)": "Temperature",
        "Humidity(%)": "Humidity",
        "Wind speed (m/s)": "WindSpeed",
        "Visibility (10m)": "Visibility",
        "Solar Radiation (MJ/m2)": "Solar_Rad",
        "Snowfall (cm)": "Snowfall",
        "Dew point temperature(°C)": "DewPtemperature",
        "Rainfall(mm)": "Rainfall",
    })
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.day_name()
    df = pd.get_dummies(df, columns=["Seasons", "Holiday", "Functioning Day"], drop_first=True)
    df["Day_Type"] = np.where(
        (df["Weekday"].isin(["Saturday", "Sunday"])) | (df["Holiday_No Holiday"] == 0),
        "Leisure",
        "Work",
    )
    raw_df = df.copy()
    df.drop(["Date", "DewPtemperature", "Weekday"], axis=1, inplace=True)
    df["Day_Type"] = df["Day_Type"].map({"Work": 0, "Leisure": 1})
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    df = df.drop(["Month", "Day", "Holiday_No Holiday"], axis=1)
    X = df.drop("RentedBike", axis=1)
    y = df["RentedBike"]
    return X, y, raw_df


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    X, y, raw_df = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    metrics = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = evaluate(y_test, pred)
        trained[name] = model
    artifact = {
        "model": trained["Random Forest"],
        "model_name": "Random Forest",
        "feature_columns": list(X.columns),
        "metrics": metrics,
        "feature_importance": dict(zip(X.columns, map(float, trained["Random Forest"].feature_importances_))),
    }
    joblib.dump(artifact, MODEL_PATH)
    summary = {
        "best_model": "Random Forest",
        "metrics": metrics,
        "rows": int(raw_df.shape[0]),
        "features": int(X.shape[1]),
        "target_mean": float(y.mean()),
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
