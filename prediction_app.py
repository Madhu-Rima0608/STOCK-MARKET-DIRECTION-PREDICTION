import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# UI

st.set_page_config(layout="wide")
st.title("📈 Stock Market ML Prediction — Pro Dashboard")

tickers = st.text_input(
    "Enter Stock Tickers (comma separated)",
    "AAPL, MSFT, TSLA"
)

ticker_list = [t.strip().upper() for t in tickers.split(",")]

# FUNCTION

def build_model(ticker):

    df = yf.download(ticker, start="2015-01-01")

    if df.empty:
        st.warning(f"No data for {ticker}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()

    # FEATURES

    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
    df["MACD"] = ta.trend.MACD(close).macd()
    df["Volatility"] = close.pct_change().rolling(10).std()

    df["Tomorrow_Close"] = close.shift(-1)
    df["Target"] = (df["Tomorrow_Close"] > close).astype(int)

    df.dropna(inplace=True)

    features = ["SMA_10","SMA_50","RSI","MACD","Volatility"]

    X = df[features]
    y = df["Target"]

    split = int(len(df)*0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # MODELS

    models = {
        "LogReg": LogisticRegression(max_iter=2000),
        "SVM": SVC(probability=True),
        "GradientBoost": GradientBoostingClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=300)
    }

    tscv = TimeSeriesSplit(n_splits=5)

    rows = []
    trained = {}

    for name, model in models.items():

        cv_f1 = cross_val_score(
            model, X_train, y_train,
            cv=tscv, scoring="f1"
        ).mean()

        model.fit(X_train, y_train)
        trained[name] = model

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1]

        rows.append({
            "Model": name,
            "CV_F1": round(cv_f1,3),
            "Accuracy": round(accuracy_score(y_test,pred),3),
            "F1": round(f1_score(y_test,pred),3),
            "ROC_AUC": round(roc_auc_score(y_test,prob),3)
        })

    results = pd.DataFrame(rows).sort_values("F1", ascending=False)

    best_name = results.iloc[0]["Model"]
    best_model = trained[best_name]

    # TOMORROW PRED

    latest = scaler.transform(df[features].iloc[-1:])
    pred_next = best_model.predict(latest)[0]
    prob_up = best_model.predict_proba(latest)[0][1]
    prob_down = 1 - prob_up

    # confidence label
    diff = abs(prob_up - 0.5)
    if diff < 0.05:
        confidence = "LOW"
    elif diff < 0.15:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    # BACKTEST

    pred_test = best_model.predict(X_test)

    bt = pd.DataFrame(index=y_test.index)
    bt["Price"] = df.loc[y_test.index, "Close"]
    bt["Pred"] = pred_test

    bt["Market_Return"] = bt["Price"].pct_change()
    bt["Strategy_Return"] = bt["Market_Return"] * bt["Pred"].shift(1)
    bt.fillna(0, inplace=True)

    bt["Market_Cum"] = (1 + bt["Market_Return"]).cumprod()
    bt["Strategy_Cum"] = (1 + bt["Strategy_Return"]).cumprod()

    return {
        "df": df,
        "results": results,
        "best_model": best_model,
        "best_name": best_name,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "pred_next": pred_next,
        "confidence": confidence,
        "backtest": bt,
        "features": features
    }

# RUN

for ticker in ticker_list:

    st.divider()
    st.header(f"📌 {ticker}")

    data = build_model(ticker)

    if data is None:
        continue

    st.subheader("📊 Model Comparison")
    st.dataframe(data["results"], use_container_width=True)

    st.success(f"Best Model: {data['best_name']}")

    # Prediction Panel

    col1, col2, col3 = st.columns(3)

    with col1:
        if data["pred_next"] == 1:
            st.metric("Direction", "📈 UP")
        else:
            st.metric("Direction", "📉 DOWN")

    with col2:
        st.metric("Prob UP", round(data["prob_up"],3))

    with col3:
        st.metric("Confidence", data["confidence"])

    st.progress(float(data["prob_up"]))

    # Feature Importance

    if data["best_name"] == "RandomForest":
        imp = data["best_model"].feature_importances_
        fi = pd.Series(imp, index=data["features"]).sort_values()

        st.subheader("🔍 Feature Importance")
        st.bar_chart(fi)

    # Backtest Plot

    import matplotlib.dates as mdates

    st.subheader("📈 Strategy vs Market")

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(data["backtest"]["Market_Cum"], label="Buy & Hold")
    ax.plot(data["backtest"]["Strategy_Cum"], label="ML Strategy")

    # ---- FIX DATE AXIS ----
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))   # show every 3rd month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)
    plt.tight_layout()

    ax.legend()
    st.pyplot(fig)

