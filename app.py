import streamlit as st
import ccxt
import pandas as pd
import numpy as np

# --- Indicator functions ---
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
    # Avoid division by zero
    denom = (max_r - min_r).replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill')
    stoch = (r - min_r) / denom
    stoch_k = stoch.rolling(k).mean() * 100
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

# --- Fetch OHLCV with fallback exchanges and token pairs ---
@st.cache_data
def fetch_data(symbol: str, timeframe="1h", limit=300):
    exchanges_to_try = ["binance", "kraken", "coinbase"]
    for exchange_id in exchanges_to_try:
        exch = getattr(ccxt, exchange_id)()
        try:
            exch.load_markets()
            # Make sure the symbol exists on that exchange
            if symbol in exch.symbols:
                ohlcv = exch.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                st.success(f"✅ Data loaded for {symbol} from {exchange_id.capitalize()}")
                return df
            else:
                st.warning(f"⚠️ Symbol {symbol} not available on {exchange_id}")
        except Exception as e:
            st.warning(f"⚠️ {exchange_id.capitalize()} failed: {e}")
            continue
    st.error(f"❌ Could not fetch data for {symbol} from any exchange.")
    return pd.DataFrame()

# --- Simple backtest ---
def simple_backtest(df, start_cash=10000, position_size=0.25):
    cash = start_cash
    position = 0.0
    trades = []
    for _, row in df.iterrows():
        price = row["close"]
        if row["Buy"] and cash > 0:
            spend = cash * position_size
            qty = spend / price
            position += qty
            cash -= spend
            trades.append({"timestamp": row["timestamp"], "type": "BUY", "price": price, "qty": qty})
        elif row["Sell"] and position > 0:
            proceeds = position * price
            cash += proceeds
            trades.append({"timestamp": row["timestamp"], "type": "SELL", "price": price, "qty": position})
            position = 0.0
    final_value = cash + position * df["close"].iloc[-1] if not df.empty else cash
    return trades, final_value

# --- Streamlit UI ---
st.title("📊 Crypto Trading Dashboard")

tokens = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "VET/USDT", "LINK/USDT", "ADA/USDT", "DOGE/USDT"]
symbol = st.selectbox("Choose token", tokens)
timeframe = st.selectbox("Timeframe", ["15m","1h","4h","1d"])

df = fetch_data(symbol, timeframe)

if not df.empty:
    # Calculate indicators
    df["RSI"] = rsi(df["close"])
    df["StochK"], df["StochD"] = stochastic_rsi(df["close"])
    df["VolMA"] = df["volume"].rolling(20).mean()

    # Signals
    df["Buy"] = (df["RSI"] < 30) & (df["StochK"] > df["StochD"]) & (df["volume"] > df["VolMA"])
    df["Sell"] = (df["RSI"] > 70) | (df["StochK"] < df["StochD"])

    # Charts
    st.subheader(f"{symbol} Price")
    st.line_chart(df.set_index("timestamp")[["close"]])

    st.subheader("Relative Strength Index (RSI)")
    st.line_chart(df.set_index("timestamp")[["RSI"]])

    st.subheader("Stochastic RSI")
    st.line_chart(df.set_index("timestamp")[["StochK","StochD"]])

    # Signals Table
    st.subheader("Trading Signals")
    signals = df[["timestamp","close","RSI","StochK","StochD","Buy","Sell"]].dropna().tail(20)
    st.dataframe(signals)

    # Backtest
    st.subheader("Backtest Results")
    trades, final_value = simple_backtest(df)
    start_cash = 10000
    profit = final_value - start_cash
    st.write(f"💰 Starting cash: ${start_cash:,.2f}")
    st.write(f"📈 Final value: ${final_value:,.2f}")
    st.write(f"✅ Net profit: ${profit:,.2f}")
    st.write(f"Number of trades: {len(trades)}")

    if trades:
        st.write("Recent trades:")
        st.dataframe(pd.DataFrame(trades).tail(10))
else:
    st.warning("No data available, check token or timeframe.")
    st.stop()
