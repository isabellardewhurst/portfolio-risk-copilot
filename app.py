import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.set_page_config(
    page_title="AI Portfolio Risk Copilot",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Helper functions
# -----------------------------

def clean_ticker(ticker):
    return str(ticker).upper().strip()


def get_prices(tickers):
    """
    Download price history from Yahoo Finance.
    Handles new yfinance MultiIndex format.
    """

    try:

        raw_data = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column"
        )

        if raw_data.empty:
            return pd.DataFrame()

        # Newer yfinance often returns MultiIndex columns
        if isinstance(raw_data.columns, pd.MultiIndex):

            # level 0 = price type (Open, Close...)
            if "Close" in raw_data.columns.get_level_values(0):
                price_data = raw_data["Close"]

            elif "Adj Close" in raw_data.columns.get_level_values(0):
                price_data = raw_data["Adj Close"]

            else:
                return pd.DataFrame()

        else:

            # single ticker case
            if "Close" in raw_data.columns:
                price_data = raw_data[["Close"]]
                price_data.columns = [tickers[0]]

            elif "Adj Close" in raw_data.columns:
                price_data = raw_data[["Adj Close"]]
                price_data.columns = [tickers[0]]

            else:
                return pd.DataFrame()

        # clean column names
        price_data.columns = [
            str(col).upper().strip()
            for col in price_data.columns
        ]

        return price_data.dropna(how="all")

    except Exception as e:

        st.error(f"Yahoo Finance error: {str(e)}")
        return pd.DataFrame()


# -----------------------------
# App intro
# -----------------------------

st.title("📊 AI Portfolio Risk Copilot")

st.write(
    "Upload a portfolio CSV and instantly see concentration, volatility, drawdown, and correlation risks."
)

st.caption(
    "For educational/demo purposes only. Not financial advice."
)

# -----------------------------
# Upload file
# -----------------------------

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

# -----------------------------
# Read portfolio
# -----------------------------

portfolio = pd.read_csv(uploaded_file)

required_columns = {"ticker", "position_value"}

if not required_columns.issubset(portfolio.columns):

    st.error("CSV must contain columns: ticker and position_value")
    st.stop()

portfolio["ticker"] = portfolio["ticker"].apply(clean_ticker)

portfolio["position_value"] = pd.to_numeric(
    portfolio["position_value"],
    errors="coerce"
)

portfolio = portfolio.dropna()
portfolio = portfolio[portfolio["position_value"] > 0]

if portfolio.empty:

    st.error("No valid holdings found.")
    st.stop()

# combine duplicates
portfolio = (
    portfolio
    .groupby("ticker", as_index=False)["position_value"]
    .sum()
)

total_value = portfolio["position_value"].sum()

portfolio["weight"] = (
    portfolio["position_value"] / total_value
)

# -----------------------------
# Portfolio display
# -----------------------------

st.success("Portfolio uploaded successfully")

st.subheader("Portfolio Holdings")

st.dataframe(portfolio)

fig_weights = px.pie(
    portfolio,
    names="ticker",
    values="position_value",
    title="Portfolio Allocation"
)

st.plotly_chart(
    fig_weights,
    use_container_width=True
)

# -----------------------------
# Download market data
# -----------------------------

tickers = portfolio["ticker"].tolist()

st.subheader("Downloading market data")

with st.spinner("Downloading prices from Yahoo Finance..."):

    price_data = get_prices(tickers)

if price_data.empty:

    st.error("Could not download market data from Yahoo Finance.")
    st.stop()

st.success("Market data downloaded successfully")

st.dataframe(price_data.tail())

# -----------------------------
# Calculate returns
# -----------------------------

returns = price_data.pct_change().dropna()

if returns.empty:

    st.error("Not enough price history.")
    st.stop()

available_tickers = [
    ticker for ticker in tickers
    if ticker in returns.columns
]

returns = returns[available_tickers]

weights = portfolio.set_index("ticker")["weight"]
weights = weights[available_tickers]

portfolio_returns = returns.dot(weights)

portfolio_growth = (1 + portfolio_returns).cumprod()

# -----------------------------
# Growth chart
# -----------------------------

st.subheader("Portfolio Growth Over Time")

fig_growth = px.line(
    portfolio_growth,
    title="Growth of $1 Invested"
)

st.plotly_chart(
    fig_growth,
    use_container_width=True
)

# -----------------------------
# Risk metrics
# -----------------------------

daily_volatility = portfolio_returns.std()

annual_volatility = daily_volatility * np.sqrt(252)

running_max = portfolio_growth.cummax()

drawdown = (
    portfolio_growth - running_max
) / running_max

max_drawdown = drawdown.min()

var_95 = np.percentile(
    portfolio_returns,
    5
)

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

# -----------------------------
# Drawdown chart
# -----------------------------

st.subheader("Drawdown")

fig_drawdown = px.line(
    drawdown,
    title="Portfolio Drawdown"
)

st.plotly_chart(
    fig_drawdown,
    use_container_width=True
)

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

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

else:

    correlation_matrix = None

    st.info(
        "Need at least two holdings for correlation."
    )

# -----------------------------
# AI risk summary
# -----------------------------

st.subheader("AI-Style Risk Summary")

largest_position = portfolio.sort_values(
    "weight",
    ascending=False
).iloc[0]

risk_comments = []

# concentration
if largest_position["weight"] > 0.40:

    risk_comments.append(
        f"Portfolio concentration is high. "
        f"{largest_position['ticker']} represents "
        f"{largest_position['weight']:.1%}."
    )

elif largest_position["weight"] > 0.25:

    risk_comments.append(
        f"Portfolio has moderate concentration risk. "
        f"Largest holding is {largest_position['ticker']} "
        f"at {largest_position['weight']:.1%}."
    )

else:

    risk_comments.append(
        "Portfolio is reasonably diversified by position size."
    )

# volatility
if annual_volatility > 0.30:

    risk_comments.append(
        "Historical volatility is high."
    )

elif annual_volatility > 0.18:

    risk_comments.append(
        "Historical volatility is moderate."
    )

else:

    risk_comments.append(
        "Historical volatility is relatively low."
    )

# drawdown
if max_drawdown < -0.25:

    risk_comments.append(
        "Portfolio experienced severe drawdowns historically."
    )

elif max_drawdown < -0.15:

    risk_comments.append(
        "Portfolio experienced moderate historical drawdowns."
    )

else:

    risk_comments.append(
        "Historical drawdowns have been relatively contained."
    )

# correlation
if correlation_matrix is not None:

    avg_corr = correlation_matrix.values[
        np.triu_indices_from(
            correlation_matrix,
            k=1
        )
    ].mean()

    if avg_corr > 0.60:

        risk_comments.append(
            "Holdings are highly correlated. Diversification benefit is limited."
        )

    elif avg_corr > 0.30:

        risk_comments.append(
            "Holdings have moderate correlation."
        )

    else:

        risk_comments.append(
            "Holdings have relatively low correlation."
        )

for comment in risk_comments:

    st.write("• " + comment)
