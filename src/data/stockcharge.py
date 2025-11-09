import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

COMPANIES = {
    "TATAPOWER.NS": "Tata Power Company Ltd",
    "SERVOTECH.NS": "Servotech Power Systems Ltd",
    "EXICOM.NS": "Exicom Tele-Systems Ltd"
}

def get_ev_demand_analysis(period: str = "1mo") -> dict:
    records = []

    for ticker, name in COMPANIES.items():
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            continue

        close_now = data["Close"].iloc[-1]
        close_start = data["Close"].iloc[0]
        pct_change = ((close_now - close_start) / close_start) * 100

        if len(data) > 5:
            recent = data["Close"].iloc[-5:]
            momentum = ((recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]) * 100
        else:
            momentum = pct_change / 2

        volatility = data["Close"].pct_change().std() * 100

        records.append({
            "Company": name,
            "Ticker": ticker,
            "Current Price (INR)": round(close_now, 2),
            "Total % Change": round(pct_change, 2),
            "Short-Term Momentum %": round(momentum, 2),
            "Volatility (%)": round(volatility, 2)
        })

    if not records:
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "period": period,
            "summary_table": pd.DataFrame(),
            "demand_index": 0,
            "demand_trend": "No Data",
            "insights": "No stock data available for this period."
        }

    df = pd.DataFrame(records)

    # Safe normalization (avoids zero-division)
    def safe_norm(series):
        if series.max() == series.min():
            return np.ones(len(series)) * 0.5
        return (series - series.min()) / (series.max() - series.min())

    df["Change_norm"] = safe_norm(df["Total % Change"])
    df["Momentum_norm"] = safe_norm(df["Short-Term Momentum %"])
    df["Volatility_norm"] = 1 - safe_norm(df["Volatility (%)"])  # lower vol = stronger confidence

    # Weighted CDI per company
    df["CDI"] = (
        0.5 * df["Change_norm"] +
        0.4 * df["Momentum_norm"] +
        0.1 * df["Volatility_norm"]
    ) * 100

    # National-level aggregated CDI
    cdi_scaled = round(df["CDI"].mean(), 2)

    # Interpret demand trend
    if cdi_scaled < 25:
        trend = "Falling"
        insight = "Investor sentiment implies weakening EV charging demand."
    elif cdi_scaled < 50:
        trend = "Stable"
        insight = "Market suggests consistent but moderate demand for charging stations."
    elif cdi_scaled < 75:
        trend = "Rising"
        insight = "Stock movement indicates a growth trend in charger demand."
    else:
        trend = "Surging"
        insight = "High investor confidence suggests rapid demand expansion ahead."

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "period": period,
        "summary_table": df[[
            "Company", "Ticker", "Current Price (INR)",
            "Total % Change", "Short-Term Momentum %",
            "Volatility (%)", "CDI"
        ]],
        "demand_index": cdi_scaled,
        "demand_trend": trend,
        "insights": insight
    }
