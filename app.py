import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AI Portfolio Risk Copilot",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------
# Helper functions
# ---------------------------------

def clean_ticker(ticker):
    return str(ticker).upper().strip()


def get_prices_from_yfinance_bulk(tickers):

    try:

        st.write("Testing yfinance bulk request...")

        raw_data = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        st.write("Raw yfinance response:")
        st.write(raw_data)

        st.write("Columns:")
        st.write(raw_data.columns)

        if raw_data.empty:
            st.error("Yahoo returned EMPTY dataframe")
            return pd.DataFrame()

        if isinstance(raw_data.columns, pd.MultiIndex):

            st.write("Detected MultiIndex")

            if "Close" not in raw_data.columns.get_level_values(0):
                st.error("No Close column in MultiIndex")
                return pd.DataFrame()

            return raw_data["Close"]

        else:

            st.write("Detected normal columns")

            if "Close" not in raw_data.columns:
                st.error("No Close column found")
                return pd.DataFrame()

            return raw_data[["Close"]]

    except Exception as e:

        st.error(f"Yahoo bulk exception: {str(e)}")

        return pd.DataFrame()

def get_prices_from_yfinance_bulk(tickers):

    try:

        st.write("Testing yfinance bulk request...")

        raw_data = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        st.write("Raw yfinance response:")
        st.write(raw_data)

        st.write("Columns:")
        st.write(raw_data.columns)

        if raw_data.empty:
            st.error("Yahoo returned EMPTY dataframe")
            return pd.DataFrame()

        if isinstance(raw_data.columns, pd.MultiIndex):

            st.write("Detected MultiIndex")

            if "Close" not in raw_data.columns.get_level_values(0):
                st.error("No Close column in MultiIndex")
                return pd.DataFrame()

            return raw_data["Close"]

        else:

            st.write("Detected normal columns")

            if "Close" not in raw_data.columns:
                st.error("No Close column found")
                return pd.DataFrame()

            return raw_data[["Close"]]

    except Exception as e:

        st.error(f"Yahoo bulk exception: {str(e)}")

        return pd.DataFrame()

def get_market_prices(tickers):

    price_data = get_prices_from_yfinance_bulk(tickers)

    if not price_data.empty:
        return price_data, "Yahoo Finance bulk"

    price_data = get_prices_from_yfinance_individual(tickers)

    if not price_data.empty:
        return price_data, "Yahoo Finance individual"

    return pd.DataFrame(), "No market data source available"

    return pd.DataFrame(), "No market data source available"


# ---------------------------------
# UI
# ---------------------------------

st.title("📊 AI Portfolio Risk Copilot")

st.subheader("Connection Test")

try:
    test = yf.download(
        "AAPL",
        period="5d",
        progress=False
    )

    st.write("Yahoo test result:")

    st.dataframe(test)

except Exception as e:

    st.error(f"Yahoo direct test failed: {str(e)}")

st.write(
    "Upload a portfolio CSV and instantly see concentration, volatility, drawdown, and correlation risks."
)

uploaded_file = st.file_uploader(
    "Upload portfolio CSV",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Upload CSV with columns: ticker, position_value")

    st.code(
        """ticker,position_value
AAPL,100000
MSFT,80000
NVDA,60000
JPM,40000"""
    )

    st.stop()


portfolio = pd.read_csv(uploaded_file)

required_columns = {"ticker", "position_value"}

if not required_columns.issubset(portfolio.columns):
    st.error("CSV must contain ticker and position_value")
    st.stop()


portfolio["ticker"] = portfolio["ticker"].apply(clean_ticker)
portfolio["position_value"] = pd.to_numeric(
    portfolio["position_value"],
    errors="coerce"
)

portfolio = portfolio.dropna()
portfolio = portfolio[portfolio["position_value"] > 0]

portfolio = (
    portfolio
    .groupby("ticker", as_index=False)["position_value"]
    .sum()
)

total_value = portfolio["position_value"].sum()

portfolio["weight"] = (
    portfolio["position_value"] / total_value
)

st.success("Portfolio uploaded successfully")

st.dataframe(portfolio)

tickers = portfolio["ticker"].tolist()

st.subheader("Downloading market data")

with st.spinner("Downloading prices..."):

    price_data, data_source = get_market_prices(tickers)

if price_data.empty:

    st.error(
        "Could not download market data."
    )

    st.stop()


st.success(f"Data source used: {data_source}")

st.dataframe(price_data.tail())


returns = price_data.pct_change().dropna()

weights = portfolio.set_index("ticker")["weight"]

available_tickers = [
    t for t in tickers
    if t in returns.columns
]

returns = returns[available_tickers]

weights = weights[available_tickers]

portfolio_returns = returns.dot(weights)

portfolio_growth = (1 + portfolio_returns).cumprod()

fig = px.line(
    portfolio_growth,
    title="Portfolio Growth"
)

st.plotly_chart(fig, use_container_width=True)


daily_volatility = portfolio_returns.std()
annual_volatility = daily_volatility * np.sqrt(252)

running_max = portfolio_growth.cummax()
drawdown = (portfolio_growth - running_max) / running_max
max_drawdown = drawdown.min()

var_95 = np.percentile(portfolio_returns, 5)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Annualized Volatility",
    f"{annual_volatility:.2%}"
)

col2.metric(
    "Maximum Drawdown",
    f"{max_drawdown:.2%}"
)

col3.metric(
    "95% Daily VaR",
    f"{var_95:.2%}"
)
