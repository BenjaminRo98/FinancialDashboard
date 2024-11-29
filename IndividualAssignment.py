###################################
# Libraries 
###################################

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import urllib
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go



#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics"
                         "Holders")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret


###################################
######      DataImports      ######
###################################

# Ticker list
@st.cache_data
def fetch_ticker_list():
    return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

# Global data functions
@st.cache_data
def get_stock_data(ticker, period="1Y", interval="1d"):
    return yf.download(ticker, period=period, interval=interval)

@st.cache_data
def get_company_info(ticker):
    return YFinance(ticker).info

@st.cache_data
def get_historical_data(ticker, period="1y"):
    return yf.download(ticker, period=period, interval="1d")

@st.cache_data
def fetch_stock_data(ticker, period):
    return yf.download(ticker, period=period, interval="1d")

# Period mapping for dropdowns
PERIOD_MAPPING = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo",
    "YTD": "ytd", "1Y": "1y", "3Y": "3y",
    "5Y": "5y", "MAX": "max"
}

# Global ticker list 
ticker_list = fetch_ticker_list()

###################################
######        Sidebar        ######
###################################

def sidebar():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #B0E0E6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("FINANCIAL DASHBOARD")
    st.sidebar.image("Yahoo-Finance.png", use_column_width=True)


    # Ticker selection
    st.sidebar.header("Select Stock")
    st.session_state["ticker"] = st.sidebar.selectbox("Ticker", ticker_list)

    # Update button
    if st.sidebar.button("Update Data"):
        st.session_state["refresh"] = True

    # Sidebar navigation
    st.sidebar.header("Navigation")
    tab_selection = st.sidebar.radio(
        "Choose a Tab:",
        options=["Summary", "Chart", "Financials", "Monte Carlo Simulation", "Stock Comparison"]
    )

    # Store the selected tab in session state
    st.session_state["selected_tab"] = tab_selection

    return tab_selection


###################################
######         TAB 1         ######
###################################

def FD_tab1():
    ticker = st.session_state.get("ticker")
    company_name = yf.Ticker(ticker).info.get("longName", "Unknown Company")
    st.title("Summary")
    st.subheader(f"{company_name} ({ticker})")

    if ticker:
        info = get_company_info(ticker)

        st.write('**Company Information:**')
        st.markdown('<div style="text-align: justify;">' + \
                    info['longBusinessSummary'] + \
                    '</div><br>', unsafe_allow_html=True)

        st.subheader("Stock Price Chart")
        time_period = st.selectbox("Select Time Period:", options=PERIOD_MAPPING.keys(), index=0)
        selected_period = PERIOD_MAPPING[time_period]

        stock_data = get_stock_data(ticker, period=selected_period)
        

        min_price = stock_data['Close'].min()
        max_price = stock_data['Close'].max()


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', fill='tozeroy', fillcolor='rgba(176, 224, 230, 1.0)'))
        fig.update_layout(
            title=f"Stock Price for {company_name} ({ticker})",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis=dict(range=[min_price * 0.95, max_price * 1.05]),
            template="plotly_white",
            )
            
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key Statistics")
        key_stats = {
            'Previous Close': info.get('previousClose', 'N/A'),
            'Open': info.get('open', 'N/A'),
            'Day High': info.get('dayHigh', 'N/A'),
            'Day Low': info.get('dayLow', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Volume': info.get('volume', 'N/A'),
        }
        st.table(pd.DataFrame(key_stats.items(), columns=["Metric", "Value"]))

        st.subheader("Top Institutional Shareholder")
        stock = yf.Ticker(ticker)
        institutional_holders = stock.institutional_holders
        st.table(institutional_holders)

###################################
######         TAB 2         ######
###################################

def FD_tab2():
    ticker = st.session_state.get("ticker")
    company_name = yf.Ticker(ticker).info.get("longName", "Unknown Company")
    st.title("Stock Chart Analysis")
    st.subheader(f"{company_name} ({ticker})")
    interval_options = ["1d", "1wk", "1mo"]
    interval = st.selectbox("Select Interval", interval_options, index=0)
    stock_data = get_stock_data(ticker, period="10y", interval=interval)

    # plot type
    plot_type = st.radio("Select Plot Type", ["Line Plot", "Candle Plot"])

    # moving average
    stock_data["MA50"] = stock_data["Close"].rolling(window=50).mean()

    # volume bar colors
    volume_color = np.where(
        stock_data["Volume"].diff() > 0,
        "green",
        "red"
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]]) 
    if plot_type == "Line Plot":
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], mode="lines", name="Close Price", fill='tozeroy', fillcolor='rgba(176, 224, 230, 0.2)'), secondary_y=True)
    else: 
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
            name="Candlestick"
        ), secondary_y=True)

    fig.add_trace(go.Bar(x=stock_data.index, y=stock_data["Volume"], name="Volume", marker_color=volume_color), secondary_y=False)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["MA50"], mode="lines", name="50-Day MA"), secondary_y=True)
    fig.update_layout(
        template="plotly_white",
        height=700,
        width=1200,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
            rangeslider={"visible": False},
        ),
        yaxis2=dict(title="Stock Price", overlaying="y", side="right"),
        yaxis=dict(title="Volume"),
    )
    st.plotly_chart(fig)


###################################
######         TAB 3         ######
###################################

def FD_tab3():
    ticker = st.session_state.get("ticker")
    company_name = yf.Ticker(ticker).info.get("longName", "Unknown Company")
    st.title("Financials")
    st.subheader(f"{company_name} ({ticker})")
    statement_type = st.selectbox("Select Financial Statement:", options=["Income Statement", "Balance Sheet", "Cash Flow"], index=0)
    period_type = st.radio("Select Period:", options=["Annual", "Quarterly"])

    if ticker:
        stock = yf.Ticker(ticker)
        financial_data = {
            "Income Statement": stock.financials if period_type == "Annual" else stock.quarterly_financials,
            "Balance Sheet": stock.balance_sheet if period_type == "Annual" else stock.quarterly_balance_sheet,
            "Cash Flow": stock.cashflow if period_type == "Annual" else stock.quarterly_cashflow
        }.get(statement_type, pd.DataFrame())

        st.subheader(f"{statement_type} ({period_type})")
        flipped_data = financial_data.iloc[::-1]
        financials = flipped_data.drop('Operating Revenue', axis='rows')
        st.table(financials)


        
###################################
######         TAB 4         ######
###################################

def FD_tab4():
    ticker = st.session_state.get("ticker")
    company_name = yf.Ticker(ticker).info.get("longName", "Unknown Company")
    st.title("Monte Carlo Simulation")
    st.subheader(f"{company_name} ({ticker})")
    num_simulations = st.selectbox("Number of Simulations", [200, 500, 1000], index=0)
    time_horizon = st.selectbox("Time Horizon (days)", [30, 60, 90], index=0)
    stock_data = get_historical_data(ticker)

    if not stock_data.empty:
        daily_returns = stock_data['Close'].pct_change().dropna()
        mean_return = daily_returns.mean()
        std_dev = daily_returns.std()
        last_price = stock_data['Close'].iloc[-1]
        simulations = np.zeros((time_horizon, num_simulations))

        for sim in range(num_simulations):
            prices = [last_price]
            for _ in range(1, time_horizon):
                shock = np.random.normal(mean_return, std_dev)
                prices.append(prices[-1] * (1 + shock))
            simulations[:, sim] = prices

        st.subheader(f"{num_simulations} Simulated Stock Price Paths")
        plt.figure(figsize=(10, 6))
        plt.plot(simulations)
        plt.title("Monte Carlo Simulation")
        plt.xlabel("Days")
        plt.ylabel("Price")
        st.pyplot(plt)


###################################
######         TAB 5         ######
###################################

def FD_tab5():
    ticker = st.session_state.get("ticker")
    company_name = yf.Ticker(ticker).info.get("longName", "Unknown Company")
    st.title("Stock Comparison")
    st.subheader(f"{company_name} ({ticker})")
    
    # Dropdowns for additional stock selections, default empty
    ticker2 = st.selectbox("Select Stock 2 (Optional)", options=[""] + list(ticker_list), index=0)
    ticker3 = st.selectbox("Select Stock 3 (Optional)", options=[""] + list(ticker_list), index=0)

    # List of tickers to fetch data for, excluding empty selections
    tickers = [ticker] + [t for t in [ticker2, ticker3] if t]

    if tickers:
        stock_data = {t: fetch_stock_data(t, "1y") for t in tickers}
        

    combined_data = pd.DataFrame({
        t: data['Close'] for t, data in stock_data.items() if not data.empty
    })
    latest_prices = {
        t: data['Close'].iloc[-1] if not data.empty else None
        for t, data in stock_data.items()
        }
    st.subheader("Comparison of Selected Stocks")
    price_table = pd.DataFrame(latest_prices.items(), columns=["Stock", "Latest Price"])
    price_table["Latest Price"] = price_table["Latest Price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    st.table(price_table.style.hide(axis='index'))
    fig = go.Figure()
    for t in combined_data.columns:
        fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data[t], mode='lines', name=t))
    fig.update_layout(
        template="plotly_white",
        height=600,
        width=1000,
        xaxis_title="Date",
        yaxis_title="Stock Price"
            )
    st.plotly_chart(fig)

#==============================================================================
# Main body
#==============================================================================

selected_tab = sidebar()

if selected_tab == "Summary":
    FD_tab1()
elif selected_tab == "Chart":
    FD_tab2()
elif selected_tab == "Financials":
    FD_tab3()
elif selected_tab == "Monte Carlo Simulation":
    FD_tab4()
elif selected_tab == "Stock Comparison":
    FD_tab5()
