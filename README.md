# AI Portfolio Risk Copilot

AI Portfolio Risk Copilot is a Streamlit web app that helps investors and analysts quickly understand hidden portfolio risks.

Users upload a CSV of holdings, and the app calculates:

- Portfolio allocation
- Position concentration
- Historical volatility
- Maximum drawdown
- Daily 95% Value at Risk
- Correlation between holdings
- Plain-English risk commentary

## Why this matters

Investment teams need fast, repeatable ways to identify concentration, volatility, drawdown, and correlation risk. This app demonstrates how AI-assisted analytics can support portfolio review, risk monitoring, and analyst workflows.

## Tech stack

- Python
- Streamlit
- pandas
- NumPy
- yfinance
- Plotly

## Example CSV format

```csv
ticker,position_value
AAPL,100000
MSFT,80000
NVDA,60000
JPM,40000

Disclaimer: This app is for educational and portfolio demonstration purposes only. It is not financial advice.
