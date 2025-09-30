import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

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
        
        # FIXED: Better error handling for missing data
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if current_price is None or current_price == 0:
            if not history.empty:
                current_price = history['Close'].iloc[-1]
            else:
                raise ValueError(f"No price data available for {ticker}")
        
        data = {
            'current_price': current_price,
            'current_eps': info.get('trailingEps', 0.0),
            'forward_eps': info.get('forwardEps', 0.0),
            'dividend_per_share': info.get('dividendRate') or info.get('trailingAnnualDividendRate', 0.0),
            'beta': info.get('beta', 1.0),
            'book_value': info.get('bookValue', 0.0),
            'roe': info.get('returnOnEquity', 0.0) * 100 if info.get('returnOnEquity') else 0.0,
            'analyst_growth': info.get('earningsGrowth', 0.0) * 100 if info.get('earningsGrowth') else 0.0,
            'tax_rate': 25.0,
            'wacc': 8.0,
            'stable_growth': 3.0,
            'desired_return': 10.0,
            'years_high_growth': 5,
            'core_mos': 25.0,
            'dividend_mos': 25.0,
            'dcf_mos': 25.0,
            'ri_mos': 25.0,
            'fcf': info.get('freeCashflow', 0.0),
            'dividend_growth': 5.0,
            'monte_carlo_runs': 1000,
            'growth_adj': 10.0,
            'wacc_adj': 10.0,
            'market_cap': info.get('marketCap', 0.0)
        }
        
        # FIXED: Calculate book value from price-to-book if not available
        if data['book_value'] == 0:
            ptb = info.get('priceToBook', 0)
            if ptb > 0:
                data['book_value'] = current_price / ptb
            else:
                data['book_value'] = 20.0  # Fallback
        
        # FIXED: Calculate ROE from financials if not available
        if data['roe'] == 0:
            net_income = info.get('netIncomeToCommon', 0)
            equity = info.get('totalStockholderEquity', 0)
            if equity > 0:
                data['roe'] = (net_income / equity) * 100
        
        # Clamp values to reasonable ranges
        data['current_price'] = max(min(data['current_price'], 100000.0), 0.01)
        data['current_eps'] = max(min(data['current_eps'], 10000.0), -10000.0)  # Allow negative
        data['forward_eps'] = max(min(data['forward_eps'], 10000.0), -10000.0)  # Allow negative
        data['dividend_per_share'] = max(min(data['dividend_per_share'], 1000.0), 0.0)
        data['beta'] = max(min(data['beta'], 10.0), 0.0)
        data['book_value'] = max(min(data['book_value'], 100000.0), 0.01)
        data['roe'] = max(min(data['roe'], 200.0), -200.0)  # Allow negative ROE
        data['analyst_growth'] = max(min(data['analyst_growth'], 100.0), -50.0)  # Allow negative growth
        data['market_cap'] = max(data['market_cap'], 0.0)
        
        # FIXED: Calculate historical P/E using time-matched EPS data
        data['historical_pe'] = calculate_historical_pe(stock, history)
        data['exit_pe'] = data['historical_pe']
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}. Using defaults.")
        return get_default_data()

def calculate_historical_pe(stock, history):
    """
    FIXED: Calculate historical P/E using properly time-matched data.
    Uses quarterly earnings data matched to corresponding stock prices.
    """
    try:
        # Get quarterly earnings history
        earnings = stock.quarterly_earnings
        
        if earnings is None or earnings.empty:
            # Fallback to annual earnings
            earnings = stock.earnings
        
        if earnings is None or earnings.empty or history.empty:
            return 15.0  # Default P/E
        
        # Calculate P/E for each period where we have both price and EPS
        pe_ratios = []
        
        for date, row in earnings.iterrows():
            eps = row.get('Earnings', 0)
            if eps <= 0:
                continue
            
            # Find closest price date
            if date in history.index:
                price = history.loc[date, 'Close']
            else:
                # Find nearest date within 30 days
                nearby_dates = history.index[abs(history.index - date) <= pd.Timedelta(days=30)]
                if len(nearby_dates) > 0:
                    closest_date = nearby_dates[0]
                    price = history.loc[closest_date, 'Close']
                else:
                    continue
            
            pe = price / eps
            # Filter out unreasonable P/E ratios
            if 0 < pe < 200:
                pe_ratios.append(pe)
        
        if len(pe_ratios) > 0:
            # Use median to avoid outlier impact
            historical_pe = float(pd.Series(pe_ratios).median())
            return max(min(historical_pe, 100.0), 1.0)
        else:
            return 15.0
            
    except Exception as e:
        print(f"Error calculating historical P/E: {str(e)}")
        return 15.0

def get_default_data():
    """Return safe default values for testing/fallback."""
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

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_sp500_tickers():
    """
    FIXED: Robust S&P 500 fetching with multiple fallbacks.
    Returns a DataFrame with Symbol, Security (name), and GICS Sector.
    """
    # Try primary source: Wikipedia
    try:
        return fetch_sp500_from_wikipedia()
    except Exception as e1:
        st.warning(f"Wikipedia fetch failed: {str(e1)}. Trying backup source...")
        
        # FIXED: Backup source - use a reliable CSV
        try:
            return fetch_sp500_from_backup()
        except Exception as e2:
            st.error(f"Backup fetch failed: {str(e2)}. Using minimal fallback list.")
            return get_sp500_fallback()

def fetch_sp500_from_wikipedia():
    """Fetch from Wikipedia (primary source)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    if not response.text:
        raise ValueError("Empty response from Wikipedia")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # FIXED: Try multiple table identifiers
    table = soup.find('table', {'id': 'constituents'})
    if table is None:
        table = soup.find('table', {'class': 'wikitable'})
    if table is None:
        raise ValueError("Could not find S&P 500 table")
    
    tickers = []
    names = []
    sectors = []
    
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) >= 4:
            ticker = cols[0].text.strip().replace('.', '-')
            name = cols[1].text.strip()
            sector = cols[3].text.strip()
            
            # Clean ticker symbols
            ticker = ticker.replace('\n', '').split()[0]
            
            tickers.append(ticker)
            names.append(name)
            sectors.append(sector)
    
    df = pd.DataFrame({
        'Symbol': tickers,
        'Security': names,
        'GICS Sector': sectors
    })
    
    if df.empty or len(df) < 400:  # Should have ~500 companies
        raise ValueError(f"Incomplete data: only {len(df)} companies found")
    
    return df

def fetch_sp500_from_backup():
    """
    FIXED: Fetch from a reliable backup source.
    Uses a maintained GitHub repository with S&P 500 data.
    """
    backup_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    
    df = pd.read_csv(backup_url)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'Symbol': 'Symbol',
        'Name': 'Security',
        'Sector': 'GICS Sector'
    })
    
    # Clean ticker symbols
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    
    return df[['Symbol', 'Security', 'GICS Sector']]

def get_sp500_fallback():
    """
    FIXED: Enhanced fallback list with more diverse sectors.
    Minimal fallback S&P 500 list for testing.
    """
    return pd.DataFrame({
        'Symbol': [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'INTC', 'CSCO', 'AVGO', 'ADBE', 'CRM',
            # Consumer Discretionary
            'AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX',
            # Healthcare
            'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY',
            # Financials
            'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS',
            # Communication Services
            'DIS', 'NFLX', 'CMCSA', 'VZ', 'T',
            # Consumer Staples
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'PM',
            # Industrials
            'BA', 'UNP', 'HON', 'UPS', 'CAT', 'GE',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB',
            # Utilities
            'NEE', 'DUK', 'SO', 'D',
            # Real Estate
            'AMT', 'PLD', 'CCI', 'PSA',
            # Materials
            'LIN', 'APD', 'SHW', 'NEM'
        ],
        'Security': [
            # Technology
            'Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc. Class A', 'Meta Platforms Inc.',
            'NVIDIA Corporation', 'Intel Corporation', 'Cisco Systems Inc.', 'Broadcom Inc.',
            'Adobe Inc.', 'Salesforce Inc.',
            # Consumer Discretionary
            'Amazon.com Inc.', 'Tesla Inc.', 'Home Depot Inc.', 'Nike Inc.', 'McDonald\'s Corporation',
            'Starbucks Corporation', 'Lowe\'s Companies Inc.', 'TJX Companies Inc.',
            # Healthcare
            'UnitedHealth Group Inc.', 'Johnson & Johnson', 'Pfizer Inc.', 'AbbVie Inc.',
            'Thermo Fisher Scientific Inc.', 'Abbott Laboratories', 'Danaher Corporation', 'Bristol-Myers Squibb',
            # Financials
            'Berkshire Hathaway Inc. Class B', 'JPMorgan Chase & Co.', 'Visa Inc.', 'Mastercard Inc.',
            'Bank of America Corp.', 'Wells Fargo & Company', 'Goldman Sachs Group Inc.', 'Morgan Stanley',
            # Communication Services
            'Walt Disney Company', 'Netflix Inc.', 'Comcast Corporation', 'Verizon Communications Inc.',
            'AT&T Inc.',
            # Consumer Staples
            'Walmart Inc.', 'Procter & Gamble Company', 'Coca-Cola Company', 'PepsiCo Inc.',
            'Costco Wholesale Corporation', 'Philip Morris International Inc.',
            # Industrials
            'Boeing Company', 'Union Pacific Corporation', 'Honeywell International Inc.', 'United Parcel Service Inc.',
            'Caterpillar Inc.', 'General Electric Company',
            # Energy
            'Exxon Mobil Corporation', 'Chevron Corporation', 'ConocoPhillips', 'Schlumberger Limited',
            # Utilities
            'NextEra Energy Inc.', 'Duke Energy Corporation', 'Southern Company', 'Dominion Energy Inc.',
            # Real Estate
            'American Tower Corporation', 'Prologis Inc.', 'Crown Castle International Corp.', 'Public Storage',
            # Materials
            'Linde plc', 'Air Products and Chemicals Inc.', 'Sherwin-Williams Company', 'Newmont Corporation'
        ],
        'GICS Sector': [
            # Technology (10)
            'Information Technology', 'Information Technology', 'Communication Services', 'Communication Services',
            'Information Technology', 'Information Technology', 'Information Technology', 'Information Technology',
            'Information Technology', 'Information Technology',
            # Consumer Discretionary (8)
            'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary',
            'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary',
            # Healthcare (8)
            'Health Care', 'Health Care', 'Health Care', 'Health Care',
            'Health Care', 'Health Care', 'Health Care', 'Health Care',
            # Financials (8)
            'Financials', 'Financials', 'Information Technology', 'Information Technology',
            'Financials', 'Financials', 'Financials', 'Financials',
            # Communication Services (5)
            'Communication Services', 'Communication Services', 'Communication Services', 'Communication Services',
            'Communication Services',
            # Consumer Staples (6)
            'Consumer Staples', 'Consumer Staples', 'Consumer Staples', 'Consumer Staples',
            'Consumer Staples', 'Consumer Staples',
            # Industrials (6)
            'Industrials', 'Industrials', 'Industrials', 'Industrials',
            'Industrials', 'Industrials',
            # Energy (4)
            'Energy', 'Energy', 'Energy', 'Energy',
            # Utilities (4)
            'Utilities', 'Utilities', 'Utilities', 'Utilities',
            # Real Estate (4)
            'Real Estate', 'Real Estate', 'Real Estate', 'Real Estate',
            # Materials (4)
            'Materials', 'Materials', 'Materials', 'Materials'
        ]
    })

@st.cache_data(ttl=3600)
def fetch_fundamental_data(ticker):
    """
    Fetch historical fundamental data using yfinance.
    Returns a dict with DataFrames for income, balance, cashflow, dividends, and history.
    """
    try:
        ticker = ticker.replace('.', '-')
        stock = yf.Ticker(ticker)
        
        # FIXED: Try quarterly first, fall back to annual
        income = stock.quarterly_financials.T if hasattr(stock, 'quarterly_financials') else pd.DataFrame()
        balance = stock.quarterly_balance_sheet.T if hasattr(stock, 'quarterly_balance_sheet') else pd.DataFrame()
        cashflow = stock.quarterly_cashflow.T if hasattr(stock, 'quarterly_cashflow') else pd.DataFrame()
        
        # If quarterly data is empty, use annual
        if income.empty:
            income = stock.financials.T if hasattr(stock, 'financials') else pd.DataFrame()
        if balance.empty:
            balance = stock.balance_sheet.T if hasattr(stock, 'balance_sheet') else pd.DataFrame()
        if cashflow.empty:
            cashflow = stock.cashflow.T if hasattr(stock, 'cashflow') else pd.DataFrame()
        
        dividends = stock.dividends if hasattr(stock, 'dividends') else pd.Series()
        history = stock.history(period="10y")
        
        # FIXED: Adjust for stock splits
        if not history.empty and hasattr(stock, 'splits'):
            splits = stock.splits
            if not splits.empty:
                # History from yfinance is already split-adjusted
                pass  # No additional adjustment needed
        
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
