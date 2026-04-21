import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Global plot style — dark terminal aesthetic
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#8b949e',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'text.color':       '#c9d1d9',
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'font.family':      'monospace',
    'font.size':        10,
})
print("✓ Libraries loaded successfully.")

# ============================================================
#  PART 1 — INDIVIDUAL STOCK ANALYSIS
# ============================================================

# CELL 2 — Step 1: Data Collection
TICKER = 'AAPL'

raw = yf.download(TICKER, period='6mo', auto_adjust=True, progress=False)

# FIX: newer yfinance returns MultiIndex columns even for a single ticker.
# Squeeze/flatten so 'Close' is always a plain 1D Series.
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.droplevel(1)

close = raw['Close'].dropna()

# Guard: crash early with a clear message if data is still empty
if close.empty:
    raise ValueError(f"No data returned for {TICKER}. Check ticker symbol and network access.")

print(f"\n{'='*50}")
print(f"  {TICKER} — 6-Month Data Collection")
print(f"{'='*50}")
print(f"  Period:       {close.index[0].date()}  →  {close.index[-1].date()}")
print(f"  Trading Days: {len(close)}")
print(f"  Latest Close: ${float(close.iloc[-1]):.2f}")
print(f"  Period High:  ${float(close.max()):.2f}")
print(f"  Period Low:   ${float(close.min()):.2f}")
print(f"{'='*50}")

# CELL 3 — Step 2: Trend Analysis (Moving Averages)
ma20 = close.rolling(window=20).mean()
ma50 = close.rolling(window=50).mean()

current_price = float(close.iloc[-1])
current_ma20  = float(ma20.iloc[-1])
current_ma50  = float(ma50.iloc[-1])

if current_price > current_ma20 and current_ma20 > current_ma50:
    trend = "Strong Uptrend"
    trend_color = '#3fb950'
elif current_price < current_ma20 and current_ma20 < current_ma50:
    trend = "Strong Downtrend"
    trend_color = '#f85149'
else:
    trend = "Mixed Trend"
    trend_color = '#d29922'

print(f"\n{'='*50}")
print(f"  TREND ANALYSIS — {TICKER}")
print(f"{'='*50}")
print(f"  Current Price : ${current_price:.2f}")
print(f"  20-Day MA     : ${current_ma20:.2f}")
print(f"  50-Day MA     : ${current_ma50:.2f}")
print(f"  Trend Signal  : {trend}")
print(f"{'='*50}")

# CELL 4 — Step 3: RSI Calculation (14-Day)
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window=period).mean()
    loss  = -delta.clip(upper=0).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

rsi         = compute_rsi(close)
current_rsi = float(rsi.iloc[-1])

if current_rsi > 70:
    rsi_signal = "Overbought (Possible SELL Signal)"
    rsi_color  = '#f85149'
elif current_rsi < 30:
    rsi_signal = "Oversold (Possible BUY Signal)"
    rsi_color  = '#3fb950'
else:
    rsi_signal = "Neutral"
    rsi_color  = '#d29922'

print(f"\n{'='*50}")
print(f"  RSI (14-Day) — {TICKER}")
print(f"{'='*50}")
print(f"  Current RSI : {current_rsi:.2f}")
print(f"  Signal      : {rsi_signal}")
print(f"  Overbought  : RSI > 70")
print(f"  Oversold    : RSI < 30")
print(f"{'='*50}")

# CELL 5 — Step 4: Volatility (20-Day Annualized)
daily_returns = close.pct_change().dropna()
rolling_vol   = daily_returns.rolling(window=20).std() * np.sqrt(252)
current_vol   = float(rolling_vol.iloc[-1]) * 100

if current_vol > 40:
    vol_level = "HIGH"
    vol_color = '#f85149'
elif current_vol > 25:
    vol_level = "MEDIUM"
    vol_color = '#d29922'
else:
    vol_level = "LOW"
    vol_color = '#3fb950'

print(f"\n{'='*50}")
print(f"  VOLATILITY ANALYSIS — {TICKER}")
print(f"{'='*50}")
print(f"  Ann. Volatility : {current_vol:.1f}%")
print(f"  Level           : {vol_level}")
print(f"  (High >40%, Med 25-40%, Low <25%)")
print(f"{'='*50}")

# CELL 6 — Step 5: Trading Recommendation
buy_signals  = 0
sell_signals = 0

if trend == "Strong Uptrend":
    buy_signals += 2
elif trend == "Strong Downtrend":
    sell_signals += 2

if current_rsi < 30:
    buy_signals += 2
elif current_rsi > 70:
    sell_signals += 2

vol_note = f"Volatility is {vol_level} ({current_vol:.1f}%) — factor risk into position sizing."

if buy_signals >= sell_signals and buy_signals > 0:
    recommendation = "BUY"
    rec_note = f"{trend} with supportive momentum."
elif sell_signals > buy_signals:
    recommendation = "SELL / CAUTION"
    rec_note = "Bearish signals detected. Review position."
else:
    recommendation = "HOLD"
    rec_note = "Mixed signals. Wait for clearer direction."

print(f"\n{'='*50}")
print(f"  TRADING RECOMMENDATION — {TICKER}")
print(f"{'='*50}")
print(f"  Signal     : *** {recommendation} ***")
print(f"  Rationale  : {rec_note}")
print(f"  Volatility : {vol_note}")
print(f"{'='*50}")

# CELL 7 — Part 1 Charts
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 1], hspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.plot(close.index, close, color='#58a6ff', linewidth=1.5, label='Price')
ax1.plot(ma20.index,  ma20,  color='#3fb950', linewidth=1.0, label='20-Day MA', linestyle='--')
ax1.plot(ma50.index,  ma50,  color='#d29922', linewidth=1.0, label='50-Day MA', linestyle=':')
ax1.set_title(f'{TICKER} — Price & Moving Averages (6 Months)', color='#c9d1d9', pad=10)
ax1.set_ylabel('Price ($)', color='#8b949e')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.annotate(f'  {trend}', xy=(close.index[-1], current_price),
             xytext=(-120, 10), textcoords='offset points',
             color=trend_color, fontsize=9, fontweight='bold')

ax2 = fig.add_subplot(gs[1])
ax2.plot(rsi.index, rsi, color='#a371f7', linewidth=1.2, label='RSI (14)')
ax2.axhline(70, color='#f85149', linewidth=0.7, linestyle='--', alpha=0.8, label='Overbought (70)')
ax2.axhline(30, color='#3fb950', linewidth=0.7, linestyle='--', alpha=0.8, label='Oversold (30)')
ax2.fill_between(rsi.index, 70, 100, alpha=0.06, color='#f85149')
ax2.fill_between(rsi.index, 0,  30,  alpha=0.06, color='#3fb950')
ax2.set_ylim(0, 100)
ax2.set_ylabel('RSI', color='#8b949e')
ax2.set_title('RSI — Momentum', color='#c9d1d9', pad=6)
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[2])
vol_pct = rolling_vol * 100
ax3.plot(vol_pct.index, vol_pct, color='#ff7b72', linewidth=1.2, label='Ann. Volatility')
ax3.axhline(40, color='#f85149', linewidth=0.7, linestyle='--', alpha=0.7, label='High (40%)')
ax3.axhline(25, color='#d29922', linewidth=0.7, linestyle='--', alpha=0.7, label='Medium (25%)')
ax3.set_ylabel('Volatility (%)', color='#8b949e')
ax3.set_title('20-Day Annualized Volatility', color='#c9d1d9', pad=6)
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

plt.savefig('part1_stock_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Chart saved: part1_stock_analysis.png")


# ============================================================
#  PART 2 — PORTFOLIO PERFORMANCE DASHBOARD
# ============================================================

# CELL 8 — Portfolio Setup & Data Download
PORTFOLIO_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
WEIGHTS           = [0.30,    0.25,   0.20,    0.15,   0.10]
BENCHMARK_TICKER  = 'SPY'

assert abs(sum(WEIGHTS) - 1.0) < 1e-9, "Weights must sum to 1.0"

print(f"\n{'='*50}")
print(f"  PORTFOLIO COMPOSITION")
print(f"{'='*50}")
for t, w in zip(PORTFOLIO_TICKERS, WEIGHTS):
    bar = '█' * int(w * 30)
    print(f"  {t:<6}  {w:.0%}  {bar}")
print(f"{'='*50}")
print(f"  Benchmark: {BENCHMARK_TICKER}")

all_tickers = PORTFOLIO_TICKERS + [BENCHMARK_TICKER]
raw_data    = yf.download(all_tickers, period='1y', auto_adjust=True, progress=False)['Close']

# FIX: flatten MultiIndex columns produced by newer yfinance
if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.droplevel(1)

raw_data.dropna(how='all', inplace=True)

if raw_data.empty:
    raise ValueError("No portfolio data returned. Check tickers and network access.")

print(f"\n  Downloaded {len(raw_data)} trading days "
      f"({raw_data.index[0].date()} → {raw_data.index[-1].date()})")

# CELL 9 — Step 4: Return Calculations
daily_ret      = raw_data.pct_change().dropna()
port_daily_ret = daily_ret[PORTFOLIO_TICKERS]
spy_daily_ret  = daily_ret[BENCHMARK_TICKER]

weights_arr     = np.array(WEIGHTS)
portfolio_daily = (port_daily_ret * weights_arr).sum(axis=1)

portfolio_cum = (1 + portfolio_daily).cumprod()
spy_cum       = (1 + spy_daily_ret).cumprod()

stock_total_returns = {}
for ticker in PORTFOLIO_TICKERS:
    first = float(raw_data[ticker].iloc[0])
    last  = float(raw_data[ticker].iloc[-1])
    stock_total_returns[ticker] = (last - first) / first * 100

print(f"\n{'='*50}")
print(f"  INDIVIDUAL STOCK TOTAL RETURNS (1 Year)")
print(f"{'='*50}")
for t, r in stock_total_returns.items():
    sign = '+' if r >= 0 else ''
    print(f"  {t:<6}  {sign}{r:.1f}%")
print(f"{'='*50}")

# CELL 10 — Step 5: Performance Metrics
port_total_return = float((portfolio_cum.iloc[-1] - 1) * 100)
spy_total_return  = float((spy_cum.iloc[-1] - 1) * 100)
outperformance    = port_total_return - spy_total_return

TRADING_DAYS = 252
port_vol_ann = float(portfolio_daily.std() * np.sqrt(TRADING_DAYS) * 100)
spy_vol_ann  = float(spy_daily_ret.std()   * np.sqrt(TRADING_DAYS) * 100)

RISK_FREE_RATE = 0.0525
sharpe_ratio   = (float(portfolio_daily.mean()) * TRADING_DAYS - RISK_FREE_RATE) / \
                 (float(portfolio_daily.std())  * np.sqrt(TRADING_DAYS))

rolling_max  = portfolio_cum.cummax()
drawdown     = (portfolio_cum - rolling_max) / rolling_max
max_drawdown = float(drawdown.min() * 100)

print(f"\n{'='*55}")
print(f"  PORTFOLIO PERFORMANCE SUMMARY")
print(f"{'='*55}")
print(f"  {'Metric':<30} {'Portfolio':>10} {'SPY':>10}")
print(f"  {'-'*48}")
print(f"  {'Total Return':<30} {port_total_return:>9.1f}% {spy_total_return:>9.1f}%")
print(f"  {'Outperformance':<30} {outperformance:>+9.1f}%")
print(f"  {'Annualized Volatility':<30} {port_vol_ann:>9.1f}% {spy_vol_ann:>9.1f}%")
print(f"  {'Sharpe Ratio':<30} {sharpe_ratio:>10.2f}")
print(f"  {'Max Drawdown':<30} {max_drawdown:>9.1f}%")
print(f"{'='*55}")

# CELL 11 — Step 6: Interpretation
perf_verdict   = f"OUTPERFORMED SPY by {outperformance:+.1f}%" if outperformance > 0 \
                 else f"UNDERPERFORMED SPY by {abs(outperformance):.1f}%"

vol_diff = port_vol_ann - spy_vol_ann
risk_verdict = ("MORE risky than the benchmark"    if vol_diff >  5 else
                "LESS risky than the benchmark"    if vol_diff < -5 else
                "SIMILAR risk to the benchmark")

sharpe_verdict = ("HIGHLY efficient (Sharpe > 1.5)"       if sharpe_ratio > 1.5 else
                  "EFFICIENT (Sharpe > 1.0)"               if sharpe_ratio > 1.0 else
                  "MODERATELY efficient (Sharpe 0.5–1.0)"  if sharpe_ratio > 0.5 else
                  "INEFFICIENT (Sharpe < 0.5)")

print(f"\n{'='*55}")
print(f"  INTERPRETATION")
print(f"{'='*55}")
print(f"  Performance  : Portfolio {perf_verdict}")
print(f"  Risk         : Portfolio was {risk_verdict}")
print(f"  Efficiency   : Portfolio was {sharpe_verdict}")
print(f"  Max Drawdown : {max_drawdown:.1f}% (peak-to-trough decline)")
print(f"{'='*55}")

# CELL 12 — Part 2 Charts
fig2 = plt.figure(figsize=(14, 10))
fig2.patch.set_facecolor('#0d1117')
gs2  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

ax_cum = fig2.add_subplot(gs2[0, :])
ax_cum.plot(portfolio_cum.index, (portfolio_cum - 1) * 100,
            color='#3fb950', linewidth=2.0, label='Portfolio')
ax_cum.plot(spy_cum.index, (spy_cum - 1) * 100,
            color='#58a6ff', linewidth=1.5, linestyle='--', label='SPY (Benchmark)')
ax_cum.axhline(0, color='#30363d', linewidth=0.7)
port_ret_series = (portfolio_cum - 1) * 100
spy_ret_series  = (spy_cum - 1) * 100
ax_cum.fill_between(portfolio_cum.index, port_ret_series, spy_ret_series,
                    where=(port_ret_series > spy_ret_series), alpha=0.1, color='#3fb950')
ax_cum.fill_between(portfolio_cum.index, port_ret_series, spy_ret_series,
                    where=(port_ret_series < spy_ret_series), alpha=0.1, color='#f85149')
ax_cum.set_title('Cumulative Returns — Portfolio vs SPY (1 Year)', color='#c9d1d9', pad=10)
ax_cum.set_ylabel('Return (%)', color='#8b949e')
ax_cum.legend(loc='upper left', fontsize=9)
ax_cum.grid(True, alpha=0.3)

ax_bar = fig2.add_subplot(gs2[1, 0])
ret_values = list(stock_total_returns.values())
colors_bar = ['#3fb950' if r >= 0 else '#f85149' for r in ret_values]
bars = ax_bar.bar(stock_total_returns.keys(), ret_values,
                  color=colors_bar, alpha=0.85, width=0.5)
for bar, val in zip(bars, ret_values):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f'{val:+.1f}%',
                ha='center', va='bottom', fontsize=8, color='#c9d1d9')
ax_bar.axhline(spy_total_return, color='#58a6ff', linewidth=1,
               linestyle='--', label=f'SPY {spy_total_return:.1f}%')
ax_bar.set_title('Individual Stock Returns vs SPY', color='#c9d1d9', pad=8)
ax_bar.set_ylabel('Total Return (%)', color='#8b949e')
ax_bar.legend(fontsize=8)
ax_bar.grid(True, alpha=0.2, axis='y')

ax_sc = fig2.add_subplot(gs2[1, 1])
for i, ticker in enumerate(PORTFOLIO_TICKERS):
    v = float(daily_ret[ticker].std() * np.sqrt(252) * 100)
    r = stock_total_returns[ticker]
    ax_sc.scatter(v, r, s=weights_arr[i] * 800 + 60, color='#58a6ff', alpha=0.8, zorder=3)
    ax_sc.annotate(ticker, (v, r), textcoords='offset points',
                   xytext=(5, 4), fontsize=8, color='#c9d1d9')

spy_v = float(spy_daily_ret.std() * np.sqrt(252) * 100)
ax_sc.scatter(spy_v, spy_total_return, marker='*', s=200, color='#d29922', zorder=4)
ax_sc.annotate('SPY', (spy_v, spy_total_return), textcoords='offset points',
               xytext=(5, 4), fontsize=8, color='#d29922')
ax_sc.set_title('Risk vs Return (bubble size = weight)', color='#c9d1d9', pad=8)
ax_sc.set_xlabel('Annualized Volatility (%)', color='#8b949e')
ax_sc.set_ylabel('Total Return (%)', color='#8b949e')
ax_sc.grid(True, alpha=0.25)

plt.savefig('part2_portfolio_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Chart saved: part2_portfolio_dashboard.png")
