import streamlit as st
import ccxt
import pandas as pd
import numpy as np

# --- Indicators ---
def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def stochastic_rsi(close, period=14, k=3, d=3):
    r = rsi(close, period)
    min_r = r.rolling(period).min()
    max_r = r.rolling(period).max()
    stoch = (r - min_r) / (max_r - min_r)
    stoch_k = stoch.rolling(k).mean() * 100
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

# --- Fetch OHLCV ---
@st.cache_data
def fetch_data(symbol="BTC/USDT", timeframe="1h", limit=200):
    exch = ccxt.binance()
    ohlcv = exch.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# --- UI ---
st.title("📊 Crypto Trading Dashboard")
symbol = st.selectbox("Choose symbol", ["BTC/USDT","ETH/USDT","SOL/USDT"])
timeframe = st.selectbox("Timeframe", ["15m","1h","4h","1d"])

df = fetch_data(symbol, timeframe)
df["RSI"] = rsi(df["close"])
df["StochK"], df["StochD"] = stochastic_rsi(df["close"])
df["VolMA"] = df["volume"].rolling(20).mean()

st.line_chart(df.set_index("timestamp")[["close"]])
st.line_chart(df.set_index("timestamp")[["RSI"]])
st.line_chart(df.set_index("timestamp")[["StochK","StochD"]])
