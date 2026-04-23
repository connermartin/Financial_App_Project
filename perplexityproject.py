import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

STOCK = "AAPL"
PORTFOLIO = {
    "MSFT": 0.25,
    "AAPL": 0.20,
    "NVDA": 0.20,
    "JPM": 0.20,
    "XOM": 0.15,
}
BENCHMARK = "SPY"
TRADING_DAYS = 252


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


def download_close_data(symbols, period: str):
    data = yf.download(symbols, period=period, interval="1d", auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
    return close.dropna()


def analyze_stock():
    stock_df = download_close_data(STOCK, "6mo")
    if isinstance(stock_df, pd.DataFrame) and STOCK in stock_df.columns:
        stock_df = stock_df[[STOCK]].rename(columns={STOCK: "Close"})
    else:
        stock_df.columns = ["Close"]

    stock_df["MA20"] = stock_df["Close"].rolling(20).mean()
    stock_df["MA50"] = stock_df["Close"].rolling(50).mean()
    stock_df["RSI14"] = calculate_rsi(stock_df["Close"], 14)
    stock_df["Return"] = stock_df["Close"].pct_change()
    stock_df["Vol20_Ann"] = stock_df["Return"].rolling(20).std() * np.sqrt(TRADING_DAYS)

    latest = stock_df.dropna().iloc[-1]
    price = float(latest["Close"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    rsi = float(latest["RSI14"])
    vol = float(latest["Vol20_Ann"])

    trend = classify_trend(price, ma20, ma50)
    rsi_signal = classify_rsi(rsi)
    vol_class = classify_volatility(vol)
    recommendation, explanation = trading_recommendation(trend, rsi, vol_class)

    summary = pd.DataFrame([
        {
            "Stock": STOCK,
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
    ])

    stock_df.to_csv(OUTPUT_DIR / "stock_timeseries.csv")
    summary.to_csv(OUTPUT_DIR / "stock_summary.csv", index=False)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    stock_df[["Close", "MA20", "MA50"]].plot(ax=ax)
    ax.set_title(f"{STOCK} Price and Moving Averages")
    ax.set_ylabel("Price ($)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stock_moving_averages.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    stock_df["RSI14"].plot(ax=ax, color="purple")
    ax.axhline(70, color="red", linestyle="--")
    ax.axhline(30, color="green", linestyle="--")
    ax.set_title(f"{STOCK} 14-Day RSI")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stock_rsi.png", dpi=200)
    plt.close(fig)

    return summary.iloc[0].to_dict()


def analyze_portfolio():
    symbols = list(PORTFOLIO.keys()) + [BENCHMARK]
    close = download_close_data(symbols, "1y")
    returns = close.pct_change().dropna()
    weights = pd.Series(PORTFOLIO)

    portfolio_returns = returns[weights.index].mul(weights, axis=1).sum(axis=1)
    benchmark_returns = returns[BENCHMARK]

    portfolio_total_return = float((1 + portfolio_returns).prod() - 1)
    benchmark_total_return = float((1 + benchmark_returns).prod() - 1)
    outperformance = portfolio_total_return - benchmark_total_return
    portfolio_volatility = float(portfolio_returns.std() * np.sqrt(TRADING_DAYS))
    benchmark_volatility = float(benchmark_returns.std() * np.sqrt(TRADING_DAYS))
    portfolio_sharpe = float((portfolio_returns.mean() * TRADING_DAYS) / portfolio_volatility)
    benchmark_sharpe = float((benchmark_returns.mean() * TRADING_DAYS) / benchmark_volatility)

    summary = pd.DataFrame([
        {"Metric": "Total Return", "Portfolio": portfolio_total_return, "Benchmark_SPY": benchmark_total_return},
        {"Metric": "Annualized Volatility", "Portfolio": portfolio_volatility, "Benchmark_SPY": benchmark_volatility},
        {"Metric": "Sharpe Ratio", "Portfolio": portfolio_sharpe, "Benchmark_SPY": benchmark_sharpe},
        {"Metric": "Outperformance vs SPY", "Portfolio": outperformance, "Benchmark_SPY": 0.0},
    ])

    weights_df = pd.DataFrame({"Ticker": weights.index, "Weight": weights.values})
    cumulative = pd.DataFrame({
        "Portfolio": (1 + portfolio_returns).cumprod(),
        "SPY": (1 + benchmark_returns).cumprod(),
    })

    summary.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)
    weights_df.to_csv(OUTPUT_DIR / "portfolio_weights.csv", index=False)
    cumulative.to_csv(OUTPUT_DIR / "portfolio_growth.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative.plot(ax=ax)
    ax.set_title("Portfolio vs SPY Growth of $1")
    ax.set_ylabel("Growth of $1")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "portfolio_vs_spy.png", dpi=200)
    plt.close(fig)

    return {
        "Portfolio Total Return": portfolio_total_return,
        "Benchmark Total Return": benchmark_total_return,
        "Outperformance": outperformance,
        "Portfolio Volatility": portfolio_volatility,
        "Benchmark Volatility": benchmark_volatility,
        "Portfolio Sharpe": portfolio_sharpe,
        "Benchmark Sharpe": benchmark_sharpe,
    }


def main():
    stock_results = analyze_stock()
    portfolio_results = analyze_portfolio()

    print("\nINDIVIDUAL STOCK ANALYSIS")
    for key, value in stock_results.items():
        print(f"{key}: {value}")

    print("\nPORTFOLIO PERFORMANCE")
    for key, value in portfolio_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
