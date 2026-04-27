import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Financial Analytics Dashboard", layout="wide")

st.title("Financial Analytics Dashboard")
st.caption("Python + Streamlit app for individual stock analysis and portfolio benchmarking using Yahoo Finance data.")

DEFAULT_PORTFOLIO = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
DEFAULT_WEIGHTS = [0.20, 0.20, 0.20, 0.20, 0.20]


def normalize_close(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                return s.iloc[:, 0].dropna()
            return s.dropna()
    if "Close" in df.columns:
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            return s.iloc[:, 0].dropna()
        return s.dropna()
    return df.squeeze().dropna()


def download_single_ticker(ticker: str, period: str) -> pd.Series:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    s = normalize_close(df)
    s.name = ticker.upper()
    return s


def download_multi_ticker(tickers: list[str], period: str) -> pd.DataFrame:
    df = yf.download(tickers, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("No portfolio data returned.")
    if isinstance(df.columns, pd.MultiIndex):
        closes = df["Close"].copy()
    else:
        closes = df[["Close"]].copy()
        closes.columns = tickers[:1]
    closes = closes.dropna(how="all")
    closes.columns = [str(c).upper() for c in closes.columns]
    return closes


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


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


def classify_vol(vol: float) -> str:
    if vol > 0.40:
        return "High"
    if vol >= 0.25:
        return "Medium"
    return "Low"


def final_recommendation(trend: str, rsi_label: str) -> tuple[str, str]:
    if trend == "Strong Uptrend" and "Oversold" in rsi_label:
        return "Buy", "Trend is strong and momentum suggests the stock may be temporarily undervalued."
    if trend == "Strong Uptrend" and "Neutral" in rsi_label:
        return "Buy", "Trend is positive and momentum is not stretched."
    if trend == "Strong Downtrend" and "Overbought" in rsi_label:
        return "Sell", "Trend is weak and momentum indicates the price may be overextended."
    if trend == "Strong Downtrend":
        return "Sell", "Trend remains negative, so downside risk may still be elevated."
    if "Overbought" in rsi_label:
        return "Hold", "Momentum looks stretched even though the broader trend is not decisively bearish."
    if "Oversold" in rsi_label:
        return "Buy", "Momentum is weak enough to suggest a possible rebound opportunity."
    return "Hold", "Trend and momentum do not point to a clear high-conviction trade."


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    std = returns.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (excess.mean() / std) * math.sqrt(252)


with st.sidebar:
    st.header("Inputs")
    stock_ticker = st.text_input("Stock ticker", value="AAPL").upper().strip()
    st.markdown("### Portfolio")
    portfolio_tickers = []
    portfolio_weights = []
    for i in range(5):
        c1, c2 = st.columns([2, 1])
        ticker = c1.text_input(f"Ticker {i+1}", value=DEFAULT_PORTFOLIO[i]).upper().strip()
        weight = c2.number_input(f"Weight {i+1}", min_value=0.0, max_value=1.0, value=float(DEFAULT_WEIGHTS[i]), step=0.01, format="%.2f")
        portfolio_tickers.append(ticker)
        portfolio_weights.append(weight)
    benchmark = st.text_input("Benchmark ETF", value="SPY").upper().strip()
    risk_free = st.number_input("Annual risk-free rate", min_value=0.0, max_value=0.15, value=0.04, step=0.005, format="%.3f")
    run = st.button("Run analysis", type="primary")

if run:
    clean_tickers = [t for t in portfolio_tickers if t]
    if len(clean_tickers) != 5:
        st.error("Please provide exactly 5 portfolio tickers.")
        st.stop()
    if len(set(clean_tickers)) != 5:
        st.error("Portfolio tickers must be unique.")
        st.stop()
    if not np.isclose(sum(portfolio_weights), 1.0, atol=0.01):
        st.error("Portfolio weights must sum to 1.00 (within 0.01 tolerance).")
        st.stop()

    try:
        stock_close = download_single_ticker(stock_ticker, "6mo")
        portfolio_close = download_multi_ticker(clean_tickers + [benchmark], "1y")
    except Exception as e:
        st.error(f"Data download failed: {e}")
        st.stop()

    st.header("Individual Stock Analysis")
    stock_df = pd.DataFrame({"Close": stock_close})
    stock_df["20MA"] = stock_df["Close"].rolling(20).mean()
    stock_df["50MA"] = stock_df["Close"].rolling(50).mean()
    stock_df["RSI14"] = compute_rsi(stock_df["Close"], 14)
    stock_df["Return"] = stock_df["Close"].pct_change()

    current_price = float(stock_df["Close"].iloc[-1])
    ma20 = float(stock_df["20MA"].iloc[-1])
    ma50 = float(stock_df["50MA"].iloc[-1])
    rsi_now = float(stock_df["RSI14"].iloc[-1])
    vol20 = float(stock_df["Return"].tail(20).std() * np.sqrt(252))

    trend = classify_trend(current_price, ma20, ma50)
    rsi_label = classify_rsi(rsi_now)
    vol_label = classify_vol(vol20)
    rec, rec_text = final_recommendation(trend, rsi_label)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${current_price:,.2f}")
    m2.metric("20-Day MA", f"${ma20:,.2f}")
    m3.metric("50-Day MA", f"${ma50:,.2f}")
    m4.metric("14-Day RSI", f"{rsi_now:,.2f}")

    stock_summary = pd.DataFrame([
        {"Metric": "Trend", "Value": trend},
        {"Metric": "RSI Signal", "Value": rsi_label},
        {"Metric": "20-Day Annualized Volatility", "Value": f"{vol20:.2%} ({vol_label})"},
        {"Metric": "Recommendation", "Value": f"{rec} - {rec_text}"},
    ])
    st.dataframe(stock_summary, use_container_width=True, hide_index=True)

    fig_stock = go.Figure()
    fig_stock.add_trace(go.Scatter(x=stock_df.index, y=stock_df["Close"], name="Close"))
    fig_stock.add_trace(go.Scatter(x=stock_df.index, y=stock_df["20MA"], name="20-Day MA"))
    fig_stock.add_trace(go.Scatter(x=stock_df.index, y=stock_df["50MA"], name="50-Day MA"))
    fig_stock.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), title=f"{stock_ticker} Price Trend")
    st.plotly_chart(fig_stock, use_container_width=True)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock_df.index, y=stock_df["RSI14"], name="RSI", line=dict(color="#FF7F0E")))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), title=f"{stock_ticker} RSI (14)", yaxis_range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.header("Portfolio Performance Dashboard")
    closes = portfolio_close.copy().dropna()
    benchmark_close = closes[benchmark]
    asset_close = closes[clean_tickers]
    returns = asset_close.pct_change().dropna()
    bench_returns = benchmark_close.pct_change().dropna()
    common_idx = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common_idx]
    bench_returns = bench_returns.loc[common_idx]

    weights = np.array(portfolio_weights)
    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    cumulative_port = (1 + portfolio_returns).cumprod()
    cumulative_bench = (1 + bench_returns).cumprod()

    total_return = cumulative_port.iloc[-1] - 1
    benchmark_return = cumulative_bench.iloc[-1] - 1
    outperformance = total_return - benchmark_return
    port_vol = portfolio_returns.std() * np.sqrt(252)
    bench_vol = bench_returns.std() * np.sqrt(252)
    port_sharpe = sharpe_ratio(portfolio_returns, risk_free)
    bench_sharpe = sharpe_ratio(bench_returns, risk_free)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Portfolio Return", f"{total_return:.2%}")
    k2.metric("Benchmark Return", f"{benchmark_return:.2%}")
    k3.metric("Out/Underperformance", f"{outperformance:.2%}")
    k4.metric("Portfolio Volatility", f"{port_vol:.2%}")
    k5.metric("Portfolio Sharpe", f"{port_sharpe:.2f}")

    metrics_df = pd.DataFrame([
        {"Metric": "Total Return", "Portfolio": f"{total_return:.2%}", "Benchmark": f"{benchmark_return:.2%}"},
        {"Metric": "Annualized Volatility", "Portfolio": f"{port_vol:.2%}", "Benchmark": f"{bench_vol:.2%}"},
        {"Metric": "Sharpe Ratio", "Portfolio": f"{port_sharpe:.2f}", "Benchmark": f"{bench_sharpe:.2f}"},
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    cum_df = pd.DataFrame({
        "Portfolio": cumulative_port / cumulative_port.iloc[0],
        benchmark: cumulative_bench / cumulative_bench.iloc[0],
    })
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=cum_df.index, y=cum_df["Portfolio"], name="Portfolio"))
    fig_perf.add_trace(go.Scatter(x=cum_df.index, y=cum_df[benchmark], name=benchmark))
    fig_perf.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), title="Cumulative Growth of $1")
    st.plotly_chart(fig_perf, use_container_width=True)

    weights_df = pd.DataFrame({"Ticker": clean_tickers, "Weight": portfolio_weights})
    fig_weights = go.Figure(data=[go.Pie(labels=weights_df["Ticker"], values=weights_df["Weight"], hole=0.45)])
    fig_weights.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), title="Portfolio Weights")
    st.plotly_chart(fig_weights, use_container_width=True)

    st.subheader("Written Interpretation")
    interp = []
    interp.append(f"Stock analysis suggests **{rec}** because the current setup shows a **{trend.lower()}**, RSI is **{rsi_label.lower()}**, and 20-day annualized volatility is **{vol_label.lower()}** at **{vol20:.2%}**.")
    interp.append(f"The portfolio **{'outperformed' if outperformance > 0 else 'underperformed'}** the benchmark by **{abs(outperformance):.2%}** over the last year.")
    interp.append(f"The portfolio was **{'more' if port_vol > bench_vol else 'less'} risky** than the benchmark based on annualized volatility ({port_vol:.2%} vs. {bench_vol:.2%}).")
    if pd.notna(port_sharpe) and pd.notna(bench_sharpe):
        interp.append(f"On a risk-adjusted basis, the portfolio was **{'more' if port_sharpe > bench_sharpe else 'less'} efficient** than the benchmark because its Sharpe ratio was **{port_sharpe:.2f}** versus **{bench_sharpe:.2f}**.")
    for line in interp:
        st.markdown(f"- {line}")
else:
    st.info("Set your stock and portfolio inputs in the sidebar, then click **Run analysis**.")
    st.markdown(
        """
### What this app does
- Downloads real market data from Yahoo Finance with `yfinance`
- Evaluates one stock using price, moving averages, RSI, and volatility
- Builds a 5-stock portfolio and compares it to a benchmark such as SPY
- Reports total return, risk, Sharpe ratio, and a short written interpretation
        """
    )
