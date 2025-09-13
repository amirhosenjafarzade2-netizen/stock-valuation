import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    """
    Fetch stock data from Yahoo Finance for a given ticker.
    Returns a dictionary with relevant financial metrics.
    """
    try:
        ticker = ticker.replace('.', '-')
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="5y")
        
        data = {
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 0.0))),
            'current_eps': info.get('trailingEps', info.get('trailingPE', 0.0) * info.get('currentPrice', 1.0) / max(info.get('trailingPE', 1.0), 0.01) if info.get('trailingPE', 0) > 0 else 0.0),
            'forward_eps': info.get('forwardEps', 0.0),
            'dividend_per_share': info.get('dividendRate', info.get('trailingAnnualDividendRate', 0.0)),
            'beta': info.get('beta', 1.0),
            'book_value': info.get('bookValue', info.get('priceToBook', 1.0) * info.get('currentPrice', 1.0) / max(info.get('priceToBook', 0.01), 0.01) if info.get('priceToBook', 0) > 0 else 20.0),
            'roe': info.get('returnOnEquity', 0.0) * 100 if info.get('returnOnEquity', 0.0) else (info.get('netIncomeToCommon', 0.0) / max(info.get('totalStockholderEquity', 1.0), 0.01)) * 100,
            'analyst_growth': info.get('earningsGrowth', 0.0) * 100,
            'tax_rate': 25.0,
            'wacc': 8.0,
            'stable_growth': 3.0,
            'desired_return': 10.0,
            'years_high_growth': 5,
            'core_mos': 25.0,
            'dividend_mos': 25.0,
            'dcf_mos': 25.0,
            'ri_mos': 25.0,
            'fcf': 0.0,
            'dividend_growth': 5.0,
            'monte_carlo_runs': 1000,
            'growth_adj': 10.0,
            'wacc_adj': 10.0,
            'market_cap': info.get('marketCap', 0.0)
        }
        
        data['current_price'] = max(min(data['current_price'], 10000.0), 0.01)
        data['current_eps'] = max(min(data['current_eps'], 1000.0), -1000.0)
        data['forward_eps'] = max(min(data['forward_eps'], 1000.0), 0.01)
        data['dividend_per_share'] = max(min(data['dividend_per_share'], 100.0), 0.0)
        data['beta'] = max(min(data['beta'], 10.0), 0.0)
        data['book_value'] = max(min(data['book_value'], 10000.0), 0.01)
        data['roe'] = max(min(data['roe'], 100.0), -100.0)
        data['analyst_growth'] = max(min(data['analyst_growth'], 50.0), 0.0)
        data['historical_pe'] = 15.0
        data['market_cap'] = max(min(data['market_cap'], 1e12), 0.0)
        
        if not history.empty and len(history) > 0:
            avg_close = history['Close'].mean()
            if data['current_eps'] != 0:
                calculated_pe = avg_close / data['current_eps']
                data['historical_pe'] = max(min(calculated_pe, 100.0), 0.01)
            else:
                data['historical_pe'] = 15.0
        else:
            data['historical_pe'] = 15.0
        
        data['exit_pe'] = data['historical_pe']
        
        print(f"Fetched data for {ticker}: {data}")
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}. Using defaults.")
        print(f"Fetch error for {ticker}: {str(e)}")
        return {
            'current_price': 100.0,
            'current_eps': 5.0,
            'forward_eps': 5.5,
            'dividend_per_share': 1.0,
            'beta': 1.0,
            'book_value': 20.0,
            'roe': 15.0,
            'historical_pe': 15.0,
            'exit_pe': 15.0,
            'analyst_growth': 10.0,
            'tax_rate': 25.0,
            'wacc': 8.0,
            'stable_growth': 3.0,
            'desired_return': 10.0,
            'years_high_growth': 5,
            'core_mos': 25.0,
            'dividend_mos': 25.0,
            'dcf_mos': 25.0,
            'ri_mos': 25.0,
            'fcf': 0.0,
            'dividend_growth': 5.0,
            'monte_carlo_runs': 1000,
            'growth_adj': 10.0,
            'wacc_adj': 10.0,
            'market_cap': 0.0
        }

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """
    Fetch S&P 500 tickers, sectors, and names from Wikipedia.
    Returns a DataFrame with Symbol, Security (name), and GICS Sector.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        
        tickers = []
        names = []
        sectors = []
        
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 3:
                ticker = cols[0].text.strip().replace('.', '-')
                name = cols[1].text.strip()
                sector = cols[3].text.strip()
                tickers.append(ticker)
                names.append(name)
                sectors.append(sector)
        
        return pd.DataFrame({
            'Symbol': tickers,
            'Security': names,
            'GICS Sector': sectors
        })
    
    except Exception as e:
        st.error(f"Error fetching S&P 500 list: {str(e)}. Using fallback list.")
        return pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Security': ['Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.'],
            'GICS Sector': ['Information Technology', 'Information Technology', 'Communication Services']
        })

@st.cache_data(ttl=3600)  # NEW: Cache fundamental data
def fetch_fundamental_data(ticker):
    """
    Fetch historical fundamental data using yfinance.
    Returns a dict with DataFrames for income, balance, cashflow, dividends, and history.
    """
    try:
        ticker = ticker.replace('.', '-')
        stock = yf.Ticker(ticker)
        income = stock.quarterly_financials.T
        balance = stock.quarterly_balance_sheet.T
        cashflow = stock.quarterly_cashflow.T
        dividends = stock.dividends
        history = stock.history(period="10y")
        
        for df in [income, balance, cashflow]:
            if not df.empty:
                df.index = pd.to_datetime(df.index)
        
        return {
            'income': income,
            'balance': balance,
            'cashflow': cashflow,
            'dividends': dividends,
            'history': history
        }
    
    except Exception as e:
        st.error(f"Error fetching fundamental data for {ticker}: {str(e)}.")
        return {
            'income': pd.DataFrame(),
            'balance': pd.DataFrame(),
            'cashflow': pd.DataFrame(),
            'dividends': pd.Series(),
            'history': pd.DataFrame()
        }
