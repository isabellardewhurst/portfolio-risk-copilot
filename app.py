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
        raw_data = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
            timeout=20
        )

        if raw_data.empty:
            return pd.DataFrame()

        # Newer yfinance versions sometimes return MultiIndex columns
        if isinstance(raw_data.columns, pd.MultiIndex):

            if "Close" not in raw_data.columns.get_level_values(0):
                return pd.DataFrame()

            price_data = raw_data["Close"]

        else:

            if "Close" not in raw_data.columns:
                return pd.DataFrame()

            price_data = raw_data[["Close"]]

        price_data.columns = [
            str(col).upper().strip()
            for col in price_data.columns
        ]

        return price_data

    except Exception as e:
        print("Bulk download error:", e)
        return pd.DataFrame()


def get_prices_from_yfinance_individual(tickers):

    all_prices = {}

    for ticker in tickers:

        try:

            data = yf.download(
                tickers=ticker,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=20
            )

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):

                if "Close" not in data.columns.get_level_values(0):
                    continue

                all_prices[ticker] = data["Close"].squeeze()

            else:

                if "Close" not in data.columns:
                    continue

                all_prices[ticker] = data["Close"]

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue

    if not all_prices:
        return pd.DataFrame()

    price_data = pd.DataFrame(all_prices)

    price_data.columns = [
        str(col).upper().strip()
        for col in price_data.columns
    ]

    return price_data


def get_prices_from_stooq(tickers):

    all_prices = {}

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    d1 = start_date.strftime("%Y%m%d")
    d2 = end_date.strftime("%Y%m%d")

    for ticker in tickers:

        try:

            stooq_symbol = ticker.lower() + ".us"

            url = (
                "https://stooq.com/q/d/l/"
                f"?s={stooq_symbol}&d1={d1}&d2={d2}&i=d"
            )

            data = pd.read_csv(url)

            if data.empty:
                continue

            if "Close" not in data.columns:
                continue

            data["Date"] = pd.to_datetime(data["Date"])
            data = data.set_index("Date")

            all_prices[ticker] = data["Close"]

        except Exception as e:
            print(f"Stooq error {ticker}: {e}")
            continue

    if not all_prices:
        return pd.DataFrame()

    price_data = pd.DataFrame(all_prices)

    price_data.columns = [
        str(col).upper().strip()
        for col in price_data.columns
    ]

    return price_data


def get_market_prices(tickers):

    price_data = get_prices_from_yfinance_bulk(tickers)

    if not price_data.empty:
        return price_data, "Yahoo Finance bulk"

    price_data = get_prices_from_yfinance_individual(tickers)

    if not price_data.empty:
        return price_data, "Yahoo Finance individual"

    price_data = get_prices_from_stooq(tickers)

    if not price_data.empty:
        return price_data, "Stooq backup"

    return pd.DataFrame(), "No market data source available"


# ---------------------------------
# UI
# ---------------------------------

st.title("📊 AI Portfolio Risk Copilot")

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
