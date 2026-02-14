# 📈 Stock Market Direction Prediction using Machine Learning

## 🔍 Project Overview

This project builds an end-to-end Machine Learning pipeline to predict the **next-day stock price direction (UP/DOWN)** using technical indicators and multiple classification models. It includes feature engineering, time-series validation, model comparison, backtesting, and next-day probability prediction.

The system is implemented in both:

* Jupyter Notebook (analysis version)
* Streamlit Dashboard (interactive version)

---

## 🎯 Objective

To predict whether a stock’s closing price will go **UP or DOWN tomorrow** using historical price data and technical indicators.

---

## 📊 Data Source

* Historical stock price data
* Downloaded using `yfinance`
* Fields used: Close price (primary), derived indicators

---

## 🧠 Feature Engineering

Technical indicators created:

* SMA 10
* SMA 50
* RSI
* MACD
* Rolling Volatility

Target Variable:

```
Target = 1 → Tomorrow price higher than today
Target = 0 → Tomorrow price lower than today
```

---

## 🤖 Models Compared

* Logistic Regression
* Support Vector Machine
* Gradient Boosting
* Random Forest

Model selection based on:

* F1 Score
* ROC AUC
* TimeSeries Cross Validation

---

## ⏱ Validation Strategy

TimeSeriesSplit used instead of random CV to prevent data leakage and respect chronological order.

---

## 🧪 Evaluation Metrics

* Accuracy
* F1 Score
* ROC AUC
* Cross-Validation F1

---

## 🔮 Next Day Prediction

The best model generates:

* Direction prediction (UP/DOWN)
* Probability of UP move
* Probability of DOWN move
* Confidence level (Low / Medium / High)

---

## 📈 Backtesting Strategy

Trading rule:

Buy only when model predicted UP the previous day.

Compared:

* Buy & Hold returns
* ML Strategy returns

---

## 🖥 Dashboard Features (Streamlit Version)

* Multi-stock input
* Model comparison table
* Best model auto-selection
* Probability meter
* Confidence score
* Feature importance chart
* Strategy vs Market performance chart

---

## 🛠 Tech Stack

Python, Pandas, NumPy
Scikit-learn
TA (technical indicators)
Matplotlib
Streamlit
yfinance

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is not financial advice.
