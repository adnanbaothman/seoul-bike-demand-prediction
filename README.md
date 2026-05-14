---
title: Seoul Bike Demand Intelligence
emoji: 🚲
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

## 🚀 Live Demo

[Open the App](https://huggingface.co/spaces/AdnanBaothman/seoul-bike-demand-intelligence)

# 🚲 Seoul Bike Demand Intelligence

A portfolio-ready machine learning dashboard for predicting hourly bike rental demand in Seoul using weather, time, and operational factors.

## ✨ Key Features

- Interactive demand prediction simulator
- Operational insights by hour, season, and temperature
- Model comparison using the same preprocessing logic from the original notebook
- Feature importance analysis for Random Forest
- Data explorer with filters and CSV download
- Streamlit + Plotly dashboard deployed on Hugging Face Spaces

## 🏆 Model Results

The app uses the original notebook workflow: 75/25 train-test split, Linear Regression, and Random Forest with `n_estimators=100` and `random_state=42`.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Random Forest | 0.919762 | 105.168306 | 181.200165 |
| Linear Regression | 0.535393 | 328.335523 | 436.024971 |

Production model: **Random Forest**.

## 🧠 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Hugging Face Spaces

## 📁 Project Structure

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── data/
    └── SeoulBikeData.csv
```

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Portfolio Summary

This project demonstrates an end-to-end ML workflow: data preprocessing, feature engineering, model evaluation, interactive dashboard design, and cloud deployment.
