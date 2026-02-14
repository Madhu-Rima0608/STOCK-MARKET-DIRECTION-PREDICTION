import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

st.set_page_config(layout="wide")
st.title("📈 Stock Market ML Prediction — Portfolio Dashboard")

tickers = st.text_input(
    "Enter Stock Tickers (comma separated)",
    "AAPL, MSFT, TSLA"
)

ticker_list = [t.strip().upper() for t in tickers.split(",")]

def build_model(ticker):

    df = yf.download(ticker, start="2015-01-01")

    if df.empty:
        st.warning(f"No data for {ticker}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()

    # -------- FEATURES --------

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

    # -------- MODELS --------

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

    # -------- TEST METRICS --------

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]

    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    # -------- NEXT DAY --------

    latest = scaler.transform(df[features].iloc[-1:])
    pred_next = best_model.predict(latest)[0]
    prob_up = best_model.predict_proba(latest)[0][1]

    diff = abs(prob_up - 0.5)
    confidence = "LOW" if diff < 0.05 else "MEDIUM" if diff < 0.15 else "HIGH"

    # -------- BACKTEST --------

    bt = pd.DataFrame(index=y_test.index)
    bt["Price"] = df.loc[y_test.index, "Close"]
    bt["Pred"] = y_pred

    bt["Market_Return"] = bt["Price"].pct_change()
    bt["Strategy_Return"] = bt["Market_Return"] * bt["Pred"].shift(1)
    bt.fillna(0, inplace=True)

    bt["Market_Cum"] = (1 + bt["Market_Return"]).cumprod()
    bt["Strategy_Cum"] = (1 + bt["Strategy_Return"]).cumprod()

    return {
        "results": results,
        "best_model": best_model,
        "best_name": best_name,
        "prob_up": prob_up,
        "pred_next": pred_next,
        "confidence": confidence,
        "backtest": bt,
        "features": features,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc": roc,
        "f1": f1
    }

# -------- DASHBOARD LOOP --------
for ticker in ticker_list:

    st.divider()
    st.header(f"📌 {ticker}")

    data = build_model(ticker)
    if data is None:
        continue

    # -------- MODEL TABLE --------

    st.subheader("📊 Model Comparison")
    st.dataframe(data["results"], use_container_width=True)

    st.success(f"Best Model: {data['best_name']}")

    # -------- METRICS PANEL --------

    c1,c2 = st.columns(2)
    c1.metric("Test F1", round(data["f1"],3))
    c2.metric("ROC AUC", round(data["roc"],3))

    if data["roc"] > 0.75:
        st.info("Model shows strong class separation ability")

    # -------- NEXT DAY --------

    st.subheader("🔮 Tomorrow Prediction")

    col1,col2,col3 = st.columns(3)

    col1.metric("Direction", "📈 UP" if data["pred_next"]==1 else "📉 DOWN")
    col2.metric("Prob UP", round(data["prob_up"],3))
    col3.metric("Confidence", data["confidence"])

    st.progress(float(data["prob_up"]))

    # -------- CONFUSION MATRIX --------

    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(data["y_test"], data["y_pred"])
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # -------- ROC CURVE --------

    st.subheader("📉 ROC Curve")

    fpr,tpr,_ = roc_curve(data["y_test"], data["y_prob"])

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr,tpr,label="Model")
    ax_roc.plot([0,1],[0,1],"--")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # -------- FEATURE IMPORTANCE --------

    if data["best_name"] == "RandomForest":
        st.subheader("🔍 Feature Importance")
        imp = data["best_model"].feature_importances_
        st.bar_chart(pd.Series(imp, index=data["features"]))

    # -------- BACKTEST --------

    st.subheader("📈 Strategy vs Market")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(data["backtest"]["Market_Cum"], label="Buy & Hold")
    ax.plot(data["backtest"]["Strategy_Cum"], label="ML Strategy")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    ax.legend()

    st.pyplot(fig)

