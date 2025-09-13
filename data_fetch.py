import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker):
    """
    Fetch stock data from Yahoo Finance for a given ticker.
    Returns a dictionary with relevant financial metrics.
    """
    try:
        # Initialize yfinance Ticker object
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="5y")  # For historical P/E
        
        # Extract key metrics with more robust keys and fallbacks
        data = {
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 0.0))),
            'current_eps': info.get('trailingEps', info.get('trailingPE', 0.0) * info.get('currentPrice', 1.0) / info.get('trailingPE', 1.0) if info.get('trailingPE', 0) > 0 else 0.0),
            'forward_eps': info.get('forwardEps', 0.0),
            'dividend_per_share': info.get('dividendRate', info.get('trailingAnnualDividendRate', 0.0)),
            'beta': info.get('beta', 1.0),
            'book_value': info.get('bookValue', info.get('priceToBook', 1.0) * info.get('currentPrice', 1.0) / info.get('priceToBook', 1.0) if info.get('priceToBook', 0) > 0 else 20.0),
            'roe': info.get('returnOnEquity', 0.0) * 100 if info.get('returnOnEquity', 0.0) else (info.get('netIncomeToCommon', 0.0) / info.get('totalStockholderEquity', 1.0)) * 100,
            'analyst_growth': info.get('earningsGrowth', 0.0) * 100,  # Convert to %
            'tax_rate': 25.0,  # Default
            'wacc': 8.0,  # Default
            'stable_growth': 3.0,  # Default
            'desired_return': 10.0,  # Default
            'years_high_growth': 5,  # Default
            'core_mos': 25.0,
            'dividend_mos': 25.0,
            'dcf_mos': 25.0,
            'ri_mos': 25.0,
            'fcf': 0.0,  # Default, user can set
            'dividend_growth': 5.0,
            'monte_carlo_runs': 1000,
            'growth_adj': 10.0,
            'wacc_adj': 10.0
        }
        
        # Ensure positive values for key fields
        data['current_price'] = max(data['current_price'], 0.01)
        data['current_eps'] = max(data['current_eps'], 0.0)
        data['forward_eps'] = max(data['forward_eps'], 0.01)
        data['historical_pe'] = 15.0  # Will calculate below
        
        # Calculate historical average P/E more robustly
        if not history.empty and len(history) > 0:
            avg_close = history['Close'].mean()
            if data['current_eps'] > 0:
                calculated_pe = avg_close / data['current_eps']
                data['historical_pe'] = max(calculated_pe, 0.01)
            else:
                data['historical_pe'] = 15.0
        else:
            data['historical_pe'] = 15.0
        
        # Set exit P/E to historical P/E by default
        data['exit_pe'] = data['historical_pe']
        
        # Debug: Print fetched data for troubleshooting (remove in production)
        print(f"Fetched data for {ticker}: {data}")
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}. Please enter data manually.")
        print(f"Fetch error for {ticker}: {str(e)}")  # Debug log
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
            'wacc_adj': 10.0
        }

def populate_inputs(ticker):
    """
    Fetch and return data to populate Streamlit input fields.
    Called when the 'Fetch' button is clicked in app.py.
    """
    data = fetch_stock_data(ticker)
    return data
