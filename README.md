# ✈️ Flight Delay Prediction & Analytics

## Project Overview

This project predicts flight departure delay in minutes using historical flight information.

The project combines:

- Data collection
- Data cleaning
- Exploratory Data Analysis
- Feature engineering
- Regression modeling
- Model evaluation
- Model persistence
- Flask REST API
- Streamlit dashboard

---

## Machine Learning Task

This is a regression problem.

Target:

DEPARTURE_DELAY

The model predicts the estimated departure delay in minutes.

---

## Models Used

The following regression models were evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Decision Tree Regressor

KNN was excluded from the final training pipeline because
the dataset contains millions of records and KNN prediction
became computationally expensive.

---

## Features

The model uses:

- Month
- Day
- Day of week
- Airline
- Origin airport
- Destination airport
- Scheduled departure time
- Scheduled arrival time
- Scheduled flight duration
- Distance
- Cyclical time features
- Route
- Route frequency

---

## Evaluation Metrics

The project evaluates:

- MAE
- RMSE
- R²

MAE and RMSE are measured in minutes.

R² measures the proportion of target variation explained by
the model.

---

## Project Structure

```text
Flight_Delay_Prediction/
│
├── data/
├── notebooks/
├── models/
├── src/
├── app/
├── requirements.txt
├── README.md
└── .gitignore