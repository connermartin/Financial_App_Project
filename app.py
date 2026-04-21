
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------------
# User settings
# -------------------------------
stock = 'AAPL'
portfolio = ['AAPL', 'MSFT', 'NVDA', 'JNJ', 'XOM']
weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
benchmark = 'SPY'
risk_free_rate = 0.04

# -------------------------------
# Helper functions
# -------------------------------
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def classify_trend(price, ma20, ma50):
    if price > ma20 > ma50:
        return 'Strong Uptrend'
    elif price < ma20 < ma50:
        return 'Strong Downtrend'
    else:
        return 'Mixed Trend'


def classify_rsi(rsi_value):
    if rsi_value > 70:
        return 'Overbought (Possible Sell Signal)'
    elif rsi_value < 30:
        return 'Oversold (Possible Buy Signal)'
    else:
        return 'Neutral'


def classify_volatility(vol):
    if vol > 0.40:
        return 'High'
    elif vol >= 0.25:
        return 'Medium'
    else:
        return 'Low'


def trading_recommendation(trend, rsi_value):
    if trend == 'Strong Uptrend' and rsi_value < 70:
        return 'Buy', 'Price is above both moving averages and RSI is not overbought.'
    elif trend == 'Strong Downtrend' or rsi_value > 70:
        return 'Sell', 'Trend and/or RSI suggests downside risk or stretched momentum.'
    else:
        return 'Hold', 'Signals are mixed, so waiting for stronger confirmation is reasonable.'

# -------------------------------
# Part 1: Individual stock analysis
# -------------------------------
print('\n=== PART 1: INDIVIDUAL STOCK ANALYSIS ===')

stock_data = yf.download(stock, period='6mo', interval='1d', auto_adjust=True, progress=False)
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

close_prices = stock_data['Close'].dropna()
current_price = close_prices.iloc[-1]
ma20 = close_prices.rolling(20).mean().iloc[-1]
ma50 = close_prices.rolling(50).mean().iloc[-1]
trend = classify_trend(current_price, ma20, ma50)

rsi_series = calculate_rsi(close_prices, 14)
rsi_value = rsi_series.iloc[-1]
rsi_signal = classify_rsi(rsi_value)

returns = close_prices.pct_change().dropna()
annualized_volatility = returns.tail(20).std() * np.sqrt(252)
volatility_level = classify_volatility(annualized_volatility)

recommendation, explanation = trading_recommendation(trend, rsi_value)

stock_summary = pd.DataFrame({
    'Metric': [
        'Ticker',
        'Current Price',
        '20-Day Moving Average',
        '50-Day Moving Average',
        'Trend',
        '14-Day RSI',
        'RSI Signal',
        '20-Day Annualized Volatility',
        'Volatility Level',
        'Recommendation',
        'Explanation'
    ],
    'Value': [
        stock,
        round(current_price, 2),
        round(ma20, 2),
        round(ma50, 2),
        trend,
        round(rsi_value, 2),
        rsi_signal,
        round(annualized_volatility, 4),
        volatility_level,
        recommendation,
        explanation
    ]
})

print(stock_summary.to_string(index=False))

# Chart 1: Price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(close_prices.index, close_prices, label=f'{stock} Close Price', linewidth=2)
plt.plot(close_prices.index, close_prices.rolling(20).mean(), label='20-Day MA', linewidth=2)
plt.plot(close_prices.index, close_prices.rolling(50).mean(), label='50-Day MA', linewidth=2)
plt.title(f'{stock} Price and Moving Averages (Last 6 Months)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Chart 2: RSI
plt.figure(figsize=(12, 4))
plt.plot(rsi_series.index, rsi_series, label='14-Day RSI', color='purple', linewidth=2)
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title(f'{stock} Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Part 2: Portfolio performance dashboard
# -------------------------------
print('\n=== PART 2: PORTFOLIO PERFORMANCE DASHBOARD ===')

if not np.isclose(weights.sum(), 1.0):
    raise ValueError('Portfolio weights must sum to 1.00')

all_tickers = portfolio + [benchmark]
price_data = yf.download(all_tickers, period='1y', interval='1d', auto_adjust=True, progress=False)['Close']
price_data = price_data.dropna()

stock_returns = price_data[portfolio].pct_change().dropna()
benchmark_returns = price_data[benchmark].pct_change().dropna()
portfolio_returns = stock_returns.mul(weights, axis=1).sum(axis=1)

portfolio_growth = (1 + portfolio_returns).cumprod()
benchmark_growth = (1 + benchmark_returns).cumprod()

total_return = portfolio_growth.iloc[-1] - 1
benchmark_return = benchmark_growth.iloc[-1] - 1
outperformance = total_return - benchmark_return
portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
portfolio_sharpe = ((portfolio_returns.mean() * 252) - risk_free_rate) / portfolio_volatility
benchmark_sharpe = ((benchmark_returns.mean() * 252) - risk_free_rate) / benchmark_volatility

portfolio_metrics = pd.DataFrame({
    'Metric': [
        'Portfolio Total Return',
        'Benchmark Total Return',
        'Outperformance / Underperformance',
        'Portfolio Annualized Volatility',
        'Benchmark Annualized Volatility',
        'Portfolio Sharpe Ratio',
        'Benchmark Sharpe Ratio'
    ],
    'Value': [
        round(total_return, 4),
        round(benchmark_return, 4),
        round(outperformance, 4),
        round(portfolio_volatility, 4),
        round(benchmark_volatility, 4),
        round(portfolio_sharpe, 4),
        round(benchmark_sharpe, 4)
    ]
})

weights_table = pd.DataFrame({
    'Stock': portfolio,
    'Weight': weights
})

print('\nPortfolio Weights:')
print(weights_table.to_string(index=False))

print('\nPerformance Metrics:')
print(portfolio_metrics.to_string(index=False))

# Chart 3: Portfolio vs benchmark
plt.figure(figsize=(12, 6))
plt.plot(portfolio_growth.index, portfolio_growth, label='Portfolio', linewidth=2)
plt.plot(benchmark_growth.index, benchmark_growth, label=benchmark, linewidth=2)
plt.title('Portfolio vs Benchmark Growth (Last 1 Year)')
plt.xlabel('Date')
plt.ylabel('Growth of $1 Invested')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Written interpretation
# -------------------------------
print('\n=== WRITTEN INTERPRETATION ===')
print(f"Stock analysis suggests {stock} is in a {trend.lower()}.")
print(f"RSI is {round(rsi_value, 2)}, which indicates {rsi_signal.lower()} momentum.")
print(f"20-day annualized volatility is {round(annualized_volatility * 100, 2)}%, which suggests {volatility_level.lower()} volatility.")
print(f"Final stock recommendation: {recommendation}. {explanation}")

if outperformance > 0:
    print(f"The portfolio outperformed {benchmark} by {round(outperformance * 100, 2)}%.")
else:
    print(f"The portfolio underperformed {benchmark} by {round(abs(outperformance) * 100, 2)}%.")

if portfolio_volatility > benchmark_volatility:
    print('The portfolio was slightly more risky than the benchmark based on annualized volatility.')
else:
    print('The portfolio was less risky than the benchmark based on annualized volatility.')

if portfolio_sharpe > benchmark_sharpe:
    print('The portfolio was more efficient on a risk-adjusted basis because it had a higher Sharpe ratio.')
else:
    print('The benchmark was more efficient on a risk-adjusted basis because it had a higher Sharpe ratio.')
