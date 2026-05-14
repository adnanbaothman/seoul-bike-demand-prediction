## Live Demo
[Open the App](https://huggingface.co/spaces/AdnanBaothman/seoul-bike-demand-intelligence)

# Seoul Bike Demand Intelligence

A professional machine learning dashboard that predicts hourly bike rental demand using weather, time, and operational factors from the Seoul Bike Sharing Demand dataset.

## Live App

Deploy this repository as a Hugging Face Space using the **Streamlit** SDK.

## Business Problem

Bike-sharing operators often face demand imbalance: some stations run out of bikes during peak hours while others remain underused. This app helps operations teams estimate hourly demand and make better decisions about bike rebalancing, staffing, and peak-hour preparation.

## Key Features

- Interactive demand predictor for custom weather and time scenarios.
- City-level demand insights by hour, season, and temperature.
- Model comparison across Linear Regression, Random Forest, and Gradient Boosting.
- Feature importance dashboard for explaining model behavior.
- Data explorer with CSV export.

## Model Results

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 435.53 | 327.87 | 0.536 |
| Random Forest | 171.10 | 98.60 | 0.928 |
| Gradient Boosting | 256.52 | 166.53 | 0.839 |

The production model is **Random Forest Regressor** because it achieved the strongest R² and lowest error on the test set.

## Project Structure

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── SeoulBikeData.csv
└── models/
    ├── bike_demand_model.pkl
    └── metrics.json
```

## Run Locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deploy on Hugging Face Spaces

1. Create a new Space.
2. Select **Streamlit** as the SDK.
3. Upload all files in this repository.
4. Hugging Face will install `requirements.txt` and run `app.py` automatically.

## Dataset

Seoul Bike Sharing Demand Dataset from UCI/Kaggle. The target variable is `Rented Bike Count`.

## Portfolio Summary

**Seoul Bike Demand Intelligence** is an end-to-end machine learning product that turns a notebook experiment into a deployed decision-support dashboard. It demonstrates data cleaning, feature engineering, model comparison, explainability, interactive visualization, and deployment readiness.
