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


# -----------------------------
# Helper functions
# -----------------------------

def clean_ticker(ticker):
    """
    Cleans ticker text from the CSV.
    Example: ' aapl ' becomes 'AAPL'
    """
    return str(ticker).upper().strip()


def get_prices_from_yfinance_bulk(tickers):
    """
    First attempt: download all tickers together using yfinance.
    """
    raw_data = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if raw_data.empty:
        return pd.DataFrame()

    if "Close" not in raw_data.columns:
        return pd.DataFrame()

    price_data = raw_data["Close"]

    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame(name=tickers[0])

    price_data.columns = [str(col).upper().strip() for col in price_data.columns]

    return price_data


def get_prices_from_yfinance_individual(tickers):
    """
    Second attempt: download each ticker individually using yfinance.
    Sometimes individual downloads work even when bulk download fails.
    """
    all_prices = {}

    for ticker in tickers:
        try:
            data = yf.download(
                tickers=ticker,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if data.empty:
                continue

            if "Close" not in data.columns:
                continue

            all_prices[ticker] = data["Close"]

        except Exception:
            continue

    if not all_prices:
        return pd.DataFrame()

    price_data = pd.DataFrame(all_prices)
    price_data.columns = [str(col).upper().strip() for col in price_data.columns]

    return price_data


def get_prices_from_stooq(tickers):
    """
    Third attempt: backup source using Stooq's public CSV download.

    This version is mainly for US-listed tickers.
    Example:
    AAPL becomes aapl.us
    MSFT becomes msft.us
    """
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

        except Exception:
            continue

    if not all_prices:
        return pd.DataFrame()

    price_data = pd.DataFrame(all_prices)
    price_data.columns = [str(col).upper().strip() for col in price_data.columns]

    return price_data


def get_market_prices(tickers):
    """
    Main market data function.

    It tries:
    1. yfinance bulk
    2. yfinance individual
    3. Stooq backup
    """
    price_data = get_prices_from_yfinance_bulk(tickers)

    if not price_data.empty:
        return price_data, "yfinance bulk download"

    price_data = get_prices_from_yfinance_individual(tickers)

    if not price_data.empty:
        return price_data, "yfinance individual download"

    price_data = get_prices_from_stooq(tickers)

    if not price_data.empty:
        return price_data, "Stooq backup download"

    return pd.DataFrame(), "No market data source available"


# -----------------------------
# App title and intro
# -----------------------------

st.title("📊 AI Portfolio Risk Copilot")

st.write(
    "Upload a portfolio CSV and instantly see concentration, volatility, drawdown, and correlation risks."
)

st.caption(
    "Market data is retrieved using yfinance first, with a Stooq backup for US-listed tickers. "
    "This app is for educational/demo purposes only and is not financial advice."
)


# -----------------------------
# File upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your portfolio CSV",
    type=["csv"]
)


if uploaded_file is not None:
    portfolio = pd.read_csv(uploaded_file)

    st.subheader("Your Uploaded Portfolio")
    st.dataframe(portfolio)

    required_columns = {"ticker", "position_value"}

    if not required_columns.issubset(portfolio.columns):
        st.error("Your CSV must contain two columns: ticker and position_value")
        st.stop()

    # Clean uploaded data
    portfolio["ticker"] = portfolio["ticker"].apply(clean_ticker)

    portfolio["position_value"] = pd.to_numeric(
        portfolio["position_value"],
        errors="coerce"
    )

    portfolio = portfolio.dropna(subset=["ticker", "position_value"])
    portfolio = portfolio[portfolio["position_value"] > 0]

    if portfolio.empty:
        st.error("Your CSV does not contain any valid holdings.")
        st.stop()

    # Combine duplicate tickers
    portfolio = (
        portfolio
        .groupby("ticker", as_index=False)["position_value"]
        .sum()
    )

    total_value = portfolio["position_value"].sum()
    portfolio["weight"] = portfolio["position_value"] / total_value

    st.success("Portfolio uploaded successfully!")

    st.subheader("Portfolio Weights")
    st.dataframe(portfolio)

    fig_weights = px.pie(
        portfolio,
        names="ticker",
        values="position_value",
        title="Portfolio Allocation"
    )

    st.plotly_chart(fig_weights, use_container_width=True)

    tickers = portfolio["ticker"].tolist()

    # -----------------------------
    # Market data download
    # -----------------------------

    st.subheader("Downloading Market Data")

    with st.spinner("Trying to download market data..."):
        price_data, data_source = get_market_prices(tickers)

    if price_data.empty:
        st.error(
            "Market data could not be downloaded from yfinance or the backup source. "
            "Please check that your tickers are valid US-listed tickers, for example AAPL, MSFT, NVDA, JPM."
        )

        st.write("Try this sample CSV:")

        st.code(
            """ticker,position_value
AAPL,100000
MSFT,80000
NVDA,60000
JPM,40000""",
            language="csv"
        )

        st.stop()

    price_data = price_data.dropna(how="all")

    available_tickers = [
        ticker for ticker in tickers
        if ticker in price_data.columns
    ]

    missing_tickers = [
        ticker for ticker in tickers
        if ticker not in price_data.columns
    ]

    if missing_tickers:
        st.warning(
            "No usable price data was found for these tickers: "
            + ", ".join(missing_tickers)
        )

    if len(available_tickers) == 0:
        st.error("No valid ticker data was available. Please check your CSV.")
        st.stop()

    price_data = price_data[available_tickers].dropna()

    if price_data.empty:
        st.error(
            "Price data was downloaded, but after cleaning there was no usable data left."
        )
        st.stop()

    portfolio = portfolio[portfolio["ticker"].isin(available_tickers)]

    total_value = portfolio["position_value"].sum()
    portfolio["weight"] = portfolio["position_value"] / total_value

    st.success(f"Market data downloaded successfully using: {data_source}")

    st.write("Latest price data:")
    st.dataframe(price_data.tail())

    # -----------------------------
    # Returns and portfolio growth
    # -----------------------------

    returns = price_data.pct_change().dropna()

    if returns.empty:
        st.error("There was not enough price data to calculate returns.")
        st.stop()

    returns = returns[available_tickers]

    weights = portfolio.set_index("ticker")["weight"]
    weights = weights[available_tickers]

    portfolio_returns = returns.dot(weights)

    portfolio_growth = (1 + portfolio_returns).cumprod()

    st.subheader("Portfolio Growth Over Time")

    fig_growth = px.line(
        portfolio_growth,
        title="Growth of $1 Invested"
    )

    st.plotly_chart(fig_growth, use_container_width=True)

    # -----------------------------
    # Risk metrics
    # -----------------------------

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
        "Daily 95% Value at Risk",
        f"{var_95:.2%}"
    )

    # -----------------------------
    # Drawdown chart
    # -----------------------------

    st.subheader("Drawdown Over Time")

    fig_drawdown = px.line(
        drawdown,
        title="Portfolio Drawdown"
    )

    st.plotly_chart(fig_drawdown, use_container_width=True)

    # -----------------------------
    # Correlation matrix
    # -----------------------------

    st.subheader("Correlation Matrix")

    if len(available_tickers) > 1:
        correlation_matrix = returns.corr()

        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            title="Stock Correlation Heatmap"
        )

        st.plotly_chart(fig_corr, use_container_width=True)

    else:
        correlation_matrix = None
        st.info("Correlation requires at least two valid holdings.")

    # -----------------------------
    # AI-style risk summary
    # -----------------------------

    st.subheader("AI-Style Risk Summary")

    largest_position = portfolio.sort_values(
        "weight",
        ascending=False
    ).iloc[0]

    risk_comments = []

    if largest_position["weight"] > 0.4:
        risk_comments.append(
            f"The portfolio is highly concentrated. "
            f"{largest_position['ticker']} alone represents "
            f"{largest_position['weight']:.1%} of the portfolio."
        )
    elif largest_position["weight"] > 0.25:
        risk_comments.append(
            f"The portfolio has moderate concentration risk. "
            f"The largest position is {largest_position['ticker']} at "
            f"{largest_position['weight']:.1%}."
        )
    else:
        risk_comments.append(
            "The portfolio is not heavily concentrated in a single position."
        )

    if annual_volatility > 0.30:
        risk_comments.append(
            "The portfolio has high historical volatility, meaning its value has moved sharply over the past year."
        )
    elif annual_volatility > 0.18:
        risk_comments.append(
            "The portfolio has moderate historical volatility."
        )
    else:
        risk_comments.append(
            "The portfolio has relatively low historical volatility."
        )

    if max_drawdown < -0.25:
        risk_comments.append(
            "The portfolio experienced a large historical drawdown, suggesting meaningful downside risk during market stress."
        )
    elif max_drawdown < -0.15:
        risk_comments.append(
            "The portfolio experienced a moderate drawdown over the past year."
        )
    else:
        risk_comments.append(
            "The portfolio's historical drawdown has been relatively contained."
        )

    if correlation_matrix is not None:
        avg_corr = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix, k=1)
        ].mean()

        if avg_corr > 0.6:
            risk_comments.append(
                "The holdings are highly correlated, meaning diversification may be weaker than it appears."
            )
        elif avg_corr > 0.3:
            risk_comments.append(
                "The holdings have moderate correlation with each other."
            )
        else:
            risk_comments.append(
                "The holdings have relatively low correlation, which may improve diversification."
            )
    else:
        risk_comments.append(
            "Only one valid holding was available, so the portfolio has no diversification across multiple stocks."
        )

    for comment in risk_comments:
        st.write("• " + comment)

else:
    st.info("Upload a CSV file with columns: ticker, position_value")

    st.write("Example:")

    st.code(
        """ticker,position_value
AAPL,100000
MSFT,80000
NVDA,60000
JPM,40000""",
        language="csv"
    )
