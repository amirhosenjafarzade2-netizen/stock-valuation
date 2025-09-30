import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import retrying  # NEW: Added for retry logic

@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)  # Retry 3 times, wait 2s
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
        history = stock.history(period="max")  # CHANGED: Use max period for PE calc
        
        # Better error handling for missing data
        current_price = (info.get('currentPrice') or 
                        info.get('regularMarketPrice') or 
                        info.get('previousClose'))
        if current_price is None or current_price == 0:
            if not history.empty:
                current_price = history['Close'].iloc[-1]
            else:
                raise ValueError(f"No price data available for {ticker}")
        
        # CHANGED: Use None instead of 0 for missing data
        shares_outstanding = info.get('sharesOutstanding', None)
        if shares_outstanding is None:
            st.warning(f"Shares outstanding unavailable for {ticker}. Estimating.")
            market_cap = info.get('marketCap', None)
            if market_cap and current_price:
                shares_outstanding = market_cap / current_price
        
        data = {
            'current_price': current_price,
            'current_eps': info.get('trailingEps', None),
            'forward_eps': info.get('forwardEps', None),
            'dividend_per_share': info.get('dividendRate') or 
                                 info.get('trailingAnnualDividendRate', None),
            'beta': info.get('beta', 1.0),  # Default to market beta
            'book_value': info.get('bookValue', None),
            'roe': info.get('returnOnEquity', None),
            'analyst_growth': info.get('earningsGrowth', None),
            'tax_rate': 25.0,
            'wacc': 8.0,
            'stable_growth': 3.0,
            'desired_return': 10.0,
            'years_high_growth': 5,
            'core_mos': 25.0,
            'dividend_mos': 25.0,
            'dcf_mos': 25.0,
            'ri_mos': 25.0,
            'fcf': info.get('freeCashflow', None),
            'dividend_growth': 5.0,
            'monte_carlo_runs': 1000,
            'growth_adj': 10.0,
            'wacc_adj': 10.0,
            'market_cap': info.get('marketCap', None),
            'shares_outstanding': shares_outstanding
        }
        
        # Convert percentages
        if data['roe'] is not None:
            data['roe'] = data['roe'] * 100
        if data['analyst_growth'] is not None:
            data['analyst_growth'] = data['analyst_growth'] * 100
        
        # Calculate book value from price-to-book if not available
        if data['book_value'] is None:
            ptb = info.get('priceToBook', None)
            if ptb and ptb > 0:
                data['book_value'] = current_price / ptb
            else:
                data['book_value'] = None  # CHANGED: No arbitrary fallback
        
        # Calculate ROE from financials if not available
        if data['roe'] is None:
            net_income = info.get('netIncomeToCommon', None)
            equity = info.get('totalStockholderEquity', None)
            if net_income and equity and equity != 0:
                data['roe'] = (net_income / equity) * 100
            else:
                data['roe'] = None
        
        # Validate and warn on extreme values
        if data['current_eps'] and abs(data['current_eps']) > 1000:
            st.warning(f"Extreme EPS value ({data['current_eps']:.2f}) for {ticker}. Verify.")
            data['current_eps'] = None
        if data['roe'] and abs(data['roe']) > 100:
            st.warning(f"Extreme ROE ({data['roe']:.1f}%) for {ticker}. Verify.")
            data['roe'] = None
        
        # Clamp reasonable ranges, but preserve None
        if data['current_price']:
            data['current_price'] = max(min(data['current_price'], 100000.0), 0.01)
        if data['current_eps']:
            data['current_eps'] = max(min(data['current_eps'], 10000.0), -10000.0)
        if data['forward_eps']:
            data['forward_eps'] = max(min(data['forward_eps'], 10000.0), -10000.0)
        if data['dividend_per_share']:
            data['dividend_per_share'] = max(min(data['dividend_per_share'], 1000.0), 0.0)
        if data['beta']:
            data['beta'] = max(min(data['beta'], 10.0), 0.0)
        if data['book_value']:
            data['book_value'] = max(min(data['book_value'], 100000.0), 0.01)
        if data['roe']:
            data['roe'] = max(min(data['roe'], 200.0), -200.0)
        if data['analyst_growth']:
            data['analyst_growth'] = max(min(data['analyst_growth'], 100.0), -50.0)
        if data['market_cap']:
            data['market_cap'] = max(data['market_cap'], 0.0)
        
        data['historical_pe'] = calculate_historical_pe(stock, history, info)
        data['exit_pe'] = data['historical_pe']
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}. Using defaults.")
        return get_default_data()

def calculate_historical_pe(stock, history, info):
    """
    Calculate historical P/E using annualized TTM earnings aligned with prices.
    """
    try:
        quarterly_earnings = stock.quarterly_earnings
        if quarterly_earnings is None or quarterly_earnings.empty or len(quarterly_earnings) < 4:
            trailing_pe = info.get('trailingPE', None)
            return trailing_pe if trailing_pe else 15.0
        
        # Get last 4 quarters' earnings
        ttm_earnings = quarterly_earnings['Earnings'].tail(4).sum()
        if ttm_earnings == 0:
            return 15.0
        
        # Align with latest price date
        latest_date = history.index[-1]
        prices = history['Close']
        avg_price = prices.tail(252).mean() if not prices.empty else info.get('currentPrice', 1.0)
        
        return avg_price / ttm_earnings if ttm_earnings != 0 else 15.0
    
    except Exception as e:
        st.warning(f"Error calculating historical P/E for {stock.ticker}: {str(e)}. Defaulting to 15.")
        return 15.0

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """
    Fetch S&P 500 tickers from a reliable API or fallback to hardcoded list.
    """
    try:
        # NEW: Use financialmodelingprep API (requires API key) or fallback
        api_key = st.secrets.get("FMP_API_KEY", None)  # Store in Streamlit secrets
        if api_key:
            url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            tickers = pd.DataFrame([
                {'Symbol': item['symbol'], 'Security': item.get('name', ''), 
                 'GICS Sector': item.get('sector', '')}
                for item in data
            ])
            return tickers
        
        # Fallback to Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable sortable'})
        if not table:
            raise ValueError("Wikipedia table not found.")
        
        tickers = []
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 3:
                tickers.append({
                    'Symbol': cols[0].text.strip(),
                    'Security': cols[1].text.strip(),
                    'GICS Sector': cols[3].text.strip()
                })
        return pd.DataFrame(tickers)
    
    except Exception as e:
        st.error(f"Error fetching S&P 500 tickers: {str(e)}. Using fallback.")
        # Hardcoded fallback (partial list as provided)
        return pd.DataFrame({
            'Symbol': [
                'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'INTC', 'CSCO',
                'MCD', 'SBUX', 'LOW', 'TJX', 'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY',
                'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T',
                'WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'BA', 'UNP', 'HON', 'UPS', 'CAT', 'GE',
                'XOM', 'CVX', 'COP', 'SLB', 'NEE', 'DUK', 'SO', 'D', 'AMT', 'PLD', 'CCI', 'PSA',
                'LIN', 'APD', 'SHW', 'NEM'
            ],
            'GICS Sector': [
                'Information Technology', 'Information Technology', 'Communication Services', 'Communication Services',
                'Information Technology', 'Information Technology', 'Information Technology', 'Information Technology',
                'Information Technology', 'Information Technology',
                'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary',
                'Health Care', 'Health Care', 'Health Care', 'Health Care',
                'Health Care', 'Health Care', 'Health Care', 'Health Care',
                'Financials', 'Financials', 'Information Technology', 'Information Technology',
                'Financials', 'Financials', 'Financials', 'Financials',
                'Communication Services', 'Communication Services', 'Communication Services', 'Communication Services',
                'Communication Services',
                'Consumer Staples', 'Consumer Staples', 'Consumer Staples', 'Consumer Staples',
                'Consumer Staples', 'Consumer Staples',
                'Industrials', 'Industrials', 'Industrials', 'Industrials',
                'Industrials', 'Industrials',
                'Energy', 'Energy', 'Energy', 'Energy',
                'Utilities', 'Utilities', 'Utilities', 'Utilities',
                'Real Estate', 'Real Estate', 'Real Estate', 'Real Estate',
                'Materials', 'Materials', 'Materials', 'Materials'
            ]
        })

@st.cache_data(ttl=3600)
def fetch_fundamental_data(ticker):
    """
    Fetch historical fundamental data using yfinance.
    """
    try:
        ticker = ticker.replace('.', '-')
        stock = yf.Ticker(ticker)
        
        income = stock.quarterly_financials.T if hasattr(stock, 'quarterly_financials') else pd.DataFrame()
        balance = stock.quarterly_balance_sheet.T if hasattr(stock, 'quarterly_balance_sheet') else pd.DataFrame()
        cashflow = stock.quarterly_cashflow.T if hasattr(stock, 'quarterly_cashflow') else pd.DataFrame()
        
        if income.empty:
            income = stock.financials.T if hasattr(stock, 'financials') else pd.DataFrame()
        if balance.empty:
            balance = stock.balance_sheet.T if hasattr(stock, 'balance_sheet') else pd.DataFrame()
        if cashflow.empty:
            cashflow = stock.cashflow.T if hasattr(stock, 'cashflow') else pd.DataFrame()
        
        dividends = stock.dividends if hasattr(stock, 'dividends') else pd.Series()
        history = stock.history(period="10y")
        
        for df in [income, balance, cashflow]:
            if not df.empty:
                df.index = pd.to_datetime(df.index, errors='coerce')
        
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
