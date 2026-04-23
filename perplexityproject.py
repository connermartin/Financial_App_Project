import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Financial Analytics Dashboard", layout="wide")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
TRADING_DAYS = 252
DEFAULT_PORTFOLIO = {
    "MSFT": 0.25,
    "AAPL": 0.20,
    "NVDA": 0.20,
    "JPM": 0.20,
    "XOM": 0.15,
}


def calculate_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def classify_trend(price: float, ma20: float, ma50: float) -> str:
    if price > ma20 > ma50:
        return "Strong Uptrend"
    if price < ma20 < ma50:
        return "Strong Downtrend"
    return "Mixed Trend"


def classify_rsi(rsi: float) -> str:
    if rsi > 70:
        return "Overbought (Possible Sell Signal)"
    if rsi < 30:
        return "Oversold (Possible Buy Signal)"
    return "Neutral"


def classify_volatility(volatility: float) -> str:
    if volatility > 0.40:
        return "High"
    if volatility >= 0.25:
        return "Medium"
    return "Low"


def trading_recommendation(trend: str, rsi: float, vol_class: str):
    if trend == "Strong Uptrend" and rsi < 70 and vol_class != "High":
        return "Buy", "Price is above both moving averages, momentum is not yet overbought, and volatility is not extreme."
    if trend == "Strong Downtrend" or rsi > 70:
        return "Sell", "Either trend conditions are weak/negative or RSI suggests stretched momentum to the upside."
    return "Hold", "Signals are mixed, so waiting for stronger confirmation is more consistent with the rules."


@st.cache_data(show_spinner=False)
def download_close_data(symbols, period: str) -> pd.DataFrame:
    data = yf.download(
        symbols,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        else:
            return pd.DataFrame()
    else:
        if isinstance(symbols, str):
            if "Close" not in data.columns:
                return pd.DataFrame()
            close = data[["Close"]].copy()
            close.columns = [symbols]
        else:
            close_cols = [col for col in data.columns if col in list(symbols)]
            if close_cols:
                close = data[close_cols].copy()
            elif "Close" in data.columns and len(symbols) == 1:
                close = data[["Close"]].copy()
                close.columns = list(symbols)
            else:
                return pd.DataFrame()

    close = close.dropna(how="all")
    return close


def analyze_stock(stock: str):
    stock_df = download_close_data(stock, "6mo")
    if stock_df.empty or stock not in stock_df.columns:
        raise ValueError(f"No usable Yahoo Finance close-price data was returned for {stock}.")

    stock_df = stock_df[[stock]].rename(columns={stock: "Close"}).copy()
    stock_df["MA20"] = stock_df["Close"].rolling(20).mean()
    stock_df["MA50"] = stock_df["Close"].rolling(50).mean()
    stock_df["RSI14"] = calculate_rsi(stock_df["Close"], 14)
    stock_df["Return"] = stock_df["Close"].pct_change()
    stock_df["Vol20_Ann"] = stock_df["Return"].rolling(20).std() * np.sqrt(TRADING_DAYS)

    valid = stock_df.dropna(subset=["MA20", "MA50", "RSI14", "Vol20_Ann"])
    if valid.empty:
        raise ValueError(
            f"Not enough valid observations to compute indicators for {stock}. At least 50 trading days are needed for the 50-day moving average."
        )

    latest = valid.iloc[-1]
    price = float(latest["Close"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    rsi = float(latest["RSI14"])
    vol = float(latest["Vol20_Ann"])

    trend = classify_trend(price, ma20, ma50)
    rsi_signal = classify_rsi(rsi)
    vol_class = classify_volatility(vol)
    recommendation, explanation = trading_recommendation(trend, rsi, vol_class)

    summary = {
        "Current Price": round(price, 2),
        "20-Day MA": round(ma20, 2),
        "50-Day MA": round(ma50, 2),
        "Trend": trend,
        "14-Day RSI": round(rsi, 2),
        "RSI Signal": rsi_signal,
        "20-Day Annualized Volatility": round(vol, 4),
        "Volatility Class": vol_class,
        "Recommendation": recommendation,
        "Explanation": explanation,
    }

    return stock_df, summary


def analyze_portfolio(portfolio: dict, benchmark: str):
    symbols = list(portfolio.keys()) + [benchmark]
    close = download_close_data(symbols, "1y")

    if close.empty:
        raise ValueError("No usable Yahoo Finance data was returned for the portfolio or benchmark.")

    missing = [symbol for symbol in symbols if symbol not in close.columns]
    if missing:
        raise ValueError(f"Missing close-price data for: {', '.join(missing)}")

    close = close[symbols].dropna()
    if close.empty:
        raise ValueError("After aligning dates across all holdings and the benchmark, no overlapping price history remained.")

    returns = close.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough data to calculate daily returns for the portfolio.")

    weights = pd.Series(portfolio)
    portfolio_returns = returns[weights.index].mul(weights, axis=1).sum(axis=1)
    benchmark_returns = returns[benchmark]

    portfolio_total_return = float((1 + portfolio_returns).prod() - 1)
    benchmark_total_return = float((1 + benchmark_returns).prod() - 1)
    outperformance = portfolio_total_return - benchmark_total_return
    portfolio_volatility = float(portfolio_returns.std() * np.sqrt(TRADING_DAYS))
    benchmark_volatility = float(benchmark_returns.std() * np.sqrt(TRADING_DAYS))
    portfolio_sharpe = float((portfolio_returns.mean() * TRADING_DAYS) / portfolio_volatility)
    benchmark_sharpe = float((benchmark_returns.mean() * TRADING_DAYS) / benchmark_volatility)

    metrics = {
        "Portfolio Total Return": portfolio_total_return,
        "Benchmark Return": benchmark_total_return,
        "Outperformance": outperformance,
        "Portfolio Volatility": portfolio_volatility,
        "Benchmark Volatility": benchmark_volatility,
        "Portfolio Sharpe": portfolio_sharpe,
        "Benchmark Sharpe": benchmark_sharpe,
    }

    cumulative = pd.DataFrame({
        "Portfolio": (1 + portfolio_returns).cumprod(),
        benchmark: (1 + benchmark_returns).cumprod(),
    })

    return close, cumulative, metrics


def interpretation_text(stock: str, stock_summary: dict, portfolio_metrics: dict, benchmark: str) -> str:
    outperformed = portfolio_metrics["Outperformance"] > 0
    more_risky = portfolio_metrics["Portfolio Volatility"] > portfolio_metrics["Benchmark Volatility"]
    more_efficient = portfolio_metrics["Portfolio Sharpe"] > portfolio_metrics["Benchmark Sharpe"]

    return (
        f"The stock analysis for {stock} suggests a {stock_summary['Recommendation']} stance. "
        f"Trend is {stock_summary['Trend'].lower()}, RSI is {stock_summary['RSI Signal'].lower()}, "
        f"and volatility is classified as {stock_summary['Volatility Class'].lower()}.\n\n"
        f"The portfolio {'outperformed' if outperformed else 'underperformed'} {benchmark} over the last year. "
        f"It was {'more' if more_risky else 'less'} risky than the benchmark based on annualized volatility, "
        f"and it was {'more' if more_efficient else 'less'} efficient based on the Sharpe ratio."
    )


def main():
    st.title("Financial Analytics Dashboard")
    st.write("Analyze one stock and one 5-asset portfolio using Yahoo Finance data.")

    with st.sidebar:
        st.header("Inputs")
        stock = st.text_input("Individual stock ticker", value="AAPL").upper().strip()
        benchmark = st.text_input("Benchmark ticker", value="SPY").upper().strip()
        tickers_default = ", ".join(DEFAULT_PORTFOLIO.keys())
        weights_default = ", ".join(str(v) for v in DEFAULT_PORTFOLIO.values())
        tickers_input = st.text_input("Portfolio tickers (5, comma-separated)", value=tickers_default)
        weights_input = st.text_input("Portfolio weights (comma-separated, sum to 1.0)", value=weights_default)
        run_analysis = st.button("Run analysis", type="primary")

    if not run_analysis:
        st.info("Enter your tickers and weights, then click Run analysis.")
        return

    try:
        portfolio_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        portfolio_weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]

        if len(portfolio_tickers) != 5:
            raise ValueError("Please enter exactly 5 portfolio tickers.")
        if len(portfolio_weights) != 5:
            raise ValueError("Please enter exactly 5 portfolio weights.")
        if abs(sum(portfolio_weights) - 1.0) > 1e-6:
            raise ValueError("Portfolio weights must sum to 1.00.")

        portfolio = dict(zip(portfolio_tickers, portfolio_weights))

        stock_df, stock_summary = analyze_stock(stock)
        close_df, cumulative_df, portfolio_metrics = analyze_portfolio(portfolio, benchmark)

    except Exception as exc:
        st.error(f"Analysis failed: {exc}", icon="🚨")
        return

    st.header("Individual Stock Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${stock_summary['Current Price']:.2f}")
    col2.metric("20-Day MA", f"${stock_summary['20-Day MA']:.2f}")
    col3.metric("50-Day MA", f"${stock_summary['50-Day MA']:.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("14-Day RSI", f"{stock_summary['14-Day RSI']:.2f}")
    col5.metric("Trend", stock_summary["Trend"])
    col6.metric("Recommendation", stock_summary["Recommendation"])

    stock_chart = px.line(
        stock_df.reset_index(),
        x=stock_d
