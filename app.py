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

st.title("📊 AI Portfolio Risk Copilot")

st.write(
    "Upload a portfolio CSV and instantly see concentration, volatility, drawdown, and correlation risks."
)

st.caption(
    "Market data is retrieved using yfinance for educational/demo purposes. "
    "If live market data is temporarily unavailable, please try again later."
)

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

    portfolio["ticker"] = portfolio["ticker"].astype(str).str.upper().str.strip()
    portfolio["position_value"] = pd.to_numeric(
        portfolio["position_value"],
        errors="coerce"
    )

    portfolio = portfolio.dropna(subset=["ticker", "position_value"])
    portfolio = portfolio[portfolio["position_value"] > 0]

    if portfolio.empty:
        st.error("Your CSV does not contain any valid holdings.")
        st.stop()

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

    st.subheader("Downloading Market Data")

    try:
        raw_data = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if raw_data.empty:
            st.error(
                "Market data could not be downloaded. This may be a temporary data provider issue, "
                "or one of the tickers may be invalid."
            )
            st.stop()

        if "Close" not in raw_data.columns:
            st.error(
                "The downloaded data did not contain closing prices. "
                "Please try again later or check the tickers."
            )
            st.stop()

        price_data = raw_data["Close"]

        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=tickers[0])

        price_data.columns = [str(col).upper().strip() for col in price_data.columns]

        missing_tickers = [
            ticker for ticker in tickers
            if ticker not in price_data.columns
        ]

        if missing_tickers:
            st.warning(
                "No price data was found for these tickers: "
                + ", ".join(missing_tickers)
            )

        available_tickers = [
            ticker for ticker in tickers
            if ticker in price_data.columns
        ]

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

        st.success("Market data downloaded successfully.")

        st.write("Latest price data:")
        st.dataframe(price_data.tail())

    except Exception as e:
        st.error("Something went wrong while downloading market data.")
        st.write("Error details:")
        st.code(str(e))
        st.stop()

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

    st.subheader("Drawdown Over Time")

    fig_drawdown = px.line(
        drawdown,
        title="Portfolio Drawdown"
    )

    st.plotly_chart(fig_drawdown, use_container_width=True)

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
