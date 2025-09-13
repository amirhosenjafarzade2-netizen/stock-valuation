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
        
        # Extract key metrics
        data = {
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0.0)),
            'current_eps': info.get('trailingEps', 0.0),
            'forward_eps': info.get('forwardEps', 0.0),
            'dividend_per_share': info.get('dividendRate', 0.0),
            'beta': info.get('beta', 1.0),
            'book_value': info.get('bookValue', 0.0),
            'roe': info.get('returnOnEquity', 0.0) * 100,  # Convert to %
            'analyst_growth': info.get('earningsGrowth', 0.0) * 100,  # Convert to %
            'tax_rate': 25.0,  # Default, as yfinance doesn't provide
            'wacc': 8.0,  # Default, user can adjust
            'stable_growth': 3.0,  # Default, user can adjust
        }
        
        # Calculate historical average P/E
        if not history.empty and 'Close' in history and data['current_eps'] != 0:
            historical_pe = history['Close'].mean() / data['current_eps']
            data['historical_pe'] = max(historical_pe, 0.01)  # Ensure positive
        else:
            data['historical_pe'] = 15.0  # Default fallback
        
        # Set exit P/E to historical P/E by default
        data['exit_pe'] = data['historical_pe']
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}. Please enter data manually.")
        return {
            'current_price': 0.0,
            'current_eps': 0.0,
            'forward_eps': 0.0,
            'dividend_per_share': 0.0,
            'beta': 1.0,
            'book_value': 0.0,
            'roe': 0.0,
            'historical_pe': 15.0,
            'exit_pe': 15.0,
            'analyst_growth': 0.0,
            'tax_rate': 25.0,
            'wacc': 8.0,
            'stable_growth': 3.0
        }

def populate_inputs(ticker):
    """
    Fetch and return data to populate Streamlit input fields.
    Called when the 'Fetch' button is clicked in app.py.
    """
    data = fetch_stock_data(ticker)
    return data
