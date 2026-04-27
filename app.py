import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title='Financial Analytics Dashboard', layout='wide')

st.title('📈 Stock & Portfolio Analytics Dashboard')
st.markdown('Compatible with **GitHub** and **Streamlit**. Uses Yahoo Finance market data via yfinance.')

# ---------- Helpers ----------
def download_close(tickers, period='6mo'):
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers)
    return data.dropna(how='all')

def moving_avg(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def annualized_vol(returns, window=20):
    return returns.rolling(window).std() * np.sqrt(252)

def sharpe_ratio(returns, rf=0.0):
    excess = returns.mean() * 252 - rf
    vol = returns.std() * np.sqrt(252)
    return excess / vol if vol != 0 else np.nan

# ---------- Sidebar ----------
st.sidebar.header('Inputs')
stock = st.sidebar.text_input('Individual Stock Ticker', 'AAPL').upper()
portfolio_input = st.sidebar.text_input('5 Portfolio Tickers (comma separated)', 'AAPL,MSFT,GOOGL,AMZN,NVDA')
weights_input = st.sidebar.text_input('Weights sum to 1.00', '0.2,0.2,0.2,0.2,0.2')
benchmark = st.sidebar.text_input('Benchmark ETF', 'SPY').upper()

# ---------- Part 1 ----------
st.header('Part 1: Individual Stock Analysis')
try:
    df = download_close(stock, '6mo')
    s = df.iloc[:,0].dropna()
    price = s.iloc[-1]
    ma20 = moving_avg(s, 20).iloc[-1]
    ma50 = moving_avg(s, 50).iloc[-1]
    rsi14 = rsi(s, 14).iloc[-1]
    vol20 = annualized_vol(s.pct_change().dropna(), 20).iloc[-1]

    if price > ma20 > ma50:
        trend = 'Strong Uptrend'
    elif price < ma20 < ma50:
        trend = 'Strong Downtrend'
    else:
        trend = 'Mixed Trend'

    if rsi14 > 70:
        rsi_signal = 'Overbought (Possible Sell)'
    elif rsi14 < 30:
        rsi_signal = 'Oversold (Possible Buy)'
    else:
        rsi_signal = 'Neutral'

    if vol20 > 0.40:
        vol_label = 'High'
    elif vol20 >= 0.25:
        vol_label = 'Medium'
    else:
        vol_label = 'Low'

    if trend == 'Strong Uptrend' and rsi14 < 70:
        rec = 'Buy'
    elif trend == 'Strong Downtrend' or rsi14 > 70:
        rec = 'Sell'
    else:
        rec = 'Hold'

    metrics = pd.DataFrame({
        'Metric':['Current Price','20D MA','50D MA','Trend','RSI(14)','Volatility','Recommendation'],
        'Value':[round(price,2), round(ma20,2), round(ma50,2), trend, round(rsi14,2), f'{vol20:.2%} ({vol_label})', rec]
    })
    st.dataframe(metrics, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(s.index, s.values, label='Close')
    ax.plot(s.index, moving_avg(s,20), label='20 MA')
    ax.plot(s.index, moving_avg(s,50), label='50 MA')
    ax.legend()
    ax.set_title(f'{stock} Price Trend')
    st.pyplot(fig)
except Exception as e:
    st.error(f'Error in stock analysis: {e}')

# ---------- Part 2 ----------
st.header('Part 2: Portfolio Performance Dashboard')
try:
    tickers = [t.strip().upper() for t in portfolio_input.split(',')][:5]
    weights = np.array([float(x.strip()) for x in weights_input.split(',')][:5])
    weights = weights / weights.sum()

    prices = download_close(tickers + [benchmark], '1y').dropna()
    rets = prices.pct_change().dropna()

    port_rets = (rets[tickers] * weights).sum(axis=1)
    bench_rets = rets[benchmark]

    total_port = (1 + port_rets).prod() - 1
    total_bench = (1 + bench_rets).prod() - 1
    diff = total_port - total_bench
    port_vol = port_rets.std() * np.sqrt(252)
    bench_vol = bench_rets.std() * np.sqrt(252)
    port_sharpe = sharpe_ratio(port_rets)
    bench_sharpe = sharpe_ratio(bench_rets)

    summary = pd.DataFrame({
        'Metric':['Portfolio Return','Benchmark Return','Out/Underperformance','Portfolio Volatility','Benchmark Volatility','Portfolio Sharpe','Benchmark Sharpe'],
        'Value':[f'{total_port:.2%}',f'{total_bench:.2%}',f'{diff:.2%}',f'{port_vol:.2%}',f'{bench_vol:.2%}',round(port_sharpe,2),round(bench_sharpe,2)]
    })
    st.dataframe(summary, use_container_width=True)

    growth = pd.DataFrame({
        'Portfolio': (1+port_rets).cumprod(),
        benchmark: (1+bench_rets).cumprod()
    })
    st.line_chart(growth)

    st.subheader('Interpretation')
    st.write('Outperformed benchmark:' , 'Yes' if diff > 0 else 'No')
    st.write('More risky than benchmark:' , 'Yes' if port_vol > bench_vol else 'No')
    st.write('More efficient by Sharpe:' , 'Yes' if port_sharpe > bench_sharpe else 'No')
except Exception as e:
    st.error(f'Error in portfolio analysis: {e}')

st.markdown('---')
st.caption('Deploy on Streamlit Community Cloud by uploading this file and adding requirements.txt with streamlit, yfinance, pandas, numpy, matplotlib')
