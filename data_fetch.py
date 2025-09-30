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
        
        # Better error handling for missing data
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if current_price is None or current_price == 0:
            if not history.empty:
                current_price = history['Close'].iloc[-1]
            else:
                raise ValueError(f"No price data available for {ticker}")
        
        # FIXED: Get shares outstanding for proper FCF calculations
        shares_outstanding = info.get('sharesOutstanding', 0)
        
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
            'market_cap': info.get('marketCap', 0.0),
            'shares_outstanding': shares_outstanding  # NEW: Added for proper calculations
        }
        
        # Calculate book value from price-to-book if not available
        if data['book_value'] == 0:
            ptb = info.get('priceToBook', 0)
            if ptb > 0:
                data['book_value'] = current_price / ptb
            else:
                data['book_value'] = 20.0  # Fallback
        
        # Calculate ROE from financials if not available
        if data['roe'] == 0:
            net_income = info.get('netIncomeToCommon', 0)
            equity = info.get('totalStockholderEquity', 0)
            if equity > 0:
                data['roe'] = (net_income / equity) * 100
        
        # FIXED: Validate and cap values, but warn on extremes
        if abs(data['current_eps']) > 1000:
            st.warning(f"Extreme EPS value detected ({data['current_eps']:.2f}). Verify data quality.")
        if abs(data['roe']) > 100:
            st.warning(f"Extreme ROE detected ({data['roe']:.1f}%). Verify data quality.")
        
        # Reasonable clamping
        data['current_price'] = max(min(data['current_price'], 100000.0), 0.01)
        data['current_eps'] = max(min(data['current_eps'], 10000.0), -10000.0)
        data['forward_eps'] = max(min(data['forward_eps'], 10000.0), -10000.0)
        data['dividend_per_share'] = max(min(data['dividend_per_share'], 1000.0), 0.0)
        data['beta'] = max(min(data['beta'], 10.0), 0.0)
        data['book_value'] = max(min(data['book_value'], 100000.0), 0.01)
        data['roe'] = max(min(data['roe'], 200.0), -200.0)
        data['analyst_growth'] = max(min(data['analyst_growth'], 100.0), -50.0)
        data['market_cap'] = max(data['market_cap'], 0.0)
        
        # FIXED: Calculate historical P/E using TTM data
        data['historical_pe'] = calculate_historical_pe(stock, history, info)
        data['exit_pe'] = data['historical_pe']
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}. Using defaults.")
        return get_default_data()

def calculate_historical_pe(stock, history, info):
    """
    FIXED: Calculate historical P/E using properly annualized earnings.
    Uses TTM (Trailing Twelve Months) EPS matched to historical prices.
    """
    try:
        # Get quarterly earnings history
        quarterly_earnings = stock.quarterly_earnings
        
        if quarterly_earnings is None or quarterly_earnings.empty or len(quarterly_earnings) < 4:
            # Fallback: use current trailing P/E if available
            trailing_pe = info.get('trailingPE', 0)
            if trailing_pe and 0 < trailing_pe < 200:
                return float(trailing_pe)
            return 15.0
        
        # Calculate TTM EPS for each historical point
        pe_ratios = []
        
        # Sort quarterly earnings by date (most recent first)
        quarterly_earnings = quarterly_earnings.sort_index(ascending=False)
        
        # For each quarter where we have 4+ quarters of data
        for i in range(len(quarterly_earnings) - 3):
            # Get 4 consecutive quarters for TTM calculation
            ttm_quarters = quarterly_earnings.iloc[i:i+4]
            
            # FIXED: Annualize by summing 4 quarters (TTM)
            ttm_eps = ttm_quarters['Earnings'].sum()
            
            if ttm_eps <= 0:
                continue
            
            # Get the date of the most recent quarter in this TTM period
            ttm_date = ttm_quarters.index[0]
            
            # Find closest price date (within 60 days)
            if ttm_date in history.index:
                price = history.loc[ttm_date, 'Close']
            else:
                nearby_dates = history.index[abs(history.index - ttm_date) <= pd.Timedelta(days=60)]
                if len(nearby_dates) > 0:
                    closest_date = nearby_dates[0]
                    price = history.loc[closest_date, 'Close']
                else:
                    continue
            
            pe = price / ttm_eps
            
            # Filter out unreasonable P/E ratios
            if 0 < pe < 200:
                pe_ratios.append(pe)
        
        if len(pe_ratios) >= 3:  # Need at least 3 valid P/E ratios
            # Use median to avoid outlier impact
            historical_pe = float(pd.Series(pe_ratios).median())
            return max(min(historical_pe, 100.0), 1.0)
        else:
            # Fallback to trailing P/E
            trailing_pe = info.get('trailingPE', 15.0)
            return max(min(float(trailing_pe), 100.0), 1.0) if trailing_pe else 15.0
            
    except Exception as e:
        print(f"Error calculating historical P/E: {str(e)}")
        # Final fallback
        trailing_pe = info.get('trailingPE', 15.0)
        return max(min(float(trailing_pe), 100.0), 1.0) if trailing_pe else 15.0

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
        'market_cap': 0.0,
        'shares_outstanding': 0
    }

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """
    Robust S&P 500 fetching with multiple fallbacks.
    Returns a DataFrame with Symbol, Security (name), and GICS Sector.
    """
    try:
        return fetch_sp500_from_wikipedia()
    except Exception as e1:
        st.warning(f"Wikipedia fetch failed: {str(e1)}. Trying backup source...")
        
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
            
            ticker = ticker.replace('\n', '').split()[0]
            
            tickers.append(ticker)
            names.append(name)
            sectors.append(sector)
    
    df = pd.DataFrame({
        'Symbol': tickers,
        'Security': names,
        'GICS Sector': sectors
    })
    
    if df.empty or len(df) < 400:
        raise ValueError(f"Incomplete data: only {len(df)} companies found")
    
    return df

def fetch_sp500_from_backup():
    """Fetch from a reliable backup source."""
    backup_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    
    df = pd.read_csv(backup_url)
    
    df = df.rename(columns={
        'Symbol': 'Symbol',
        'Name': 'Security',
        'Sector': 'GICS Sector'
    })
    
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    
    return df[['Symbol', 'Security', 'GICS Sector']]

def get_sp500_fallback():
    """Enhanced fallback list with more diverse sectors."""
    return pd.DataFrame({
        'Symbol': [
            'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'INTC', 'CSCO', 'AVGO', 'ADBE', 'CRM',
            'AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX',
            'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY',
            'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS',
            'DIS', 'NFLX', 'CMCSA', 'VZ', 'T',
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'PM',
            'BA', 'UNP', 'HON', 'UPS', 'CAT', 'GE',
            'XOM', 'CVX', 'COP', 'SLB',
            'NEE', 'DUK', 'SO', 'D',
            'AMT', 'PLD', 'CCI', 'PSA',
            'LIN', 'APD', 'SHW', 'NEM'
        ],
        'Security': [
            'Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc. Class A', 'Meta Platforms Inc.',
            'NVIDIA Corporation', 'Intel Corporation', 'Cisco Systems Inc.', 'Broadcom Inc.',
            'Adobe Inc.', 'Salesforce Inc.',
            'Amazon.com Inc.', 'Tesla Inc.', 'Home Depot Inc.', 'Nike Inc.', 'McDonald\'s Corporation',
            'Starbucks Corporation', 'Lowe\'s Companies Inc.', 'TJX Companies Inc.',
            'UnitedHealth Group Inc.', 'Johnson & Johnson', 'Pfizer Inc.', 'AbbVie Inc.',
            'Thermo Fisher Scientific Inc.', 'Abbott Laboratories', 'Danaher Corporation', 'Bristol-Myers Squibb',
            'Berkshire Hathaway Inc. Class B', 'JPMorgan Chase & Co.', 'Visa Inc.', 'Mastercard Inc.',
            'Bank of America Corp.', 'Wells Fargo & Company', 'Goldman Sachs Group Inc.', 'Morgan Stanley',
            'Walt Disney Company', 'Netflix Inc.', 'Comcast Corporation', 'Verizon Communications Inc.',
            'AT&T Inc.',
            'Walmart Inc.', 'Procter & Gamble Company', 'Coca-Cola Company', 'PepsiCo Inc.',
            'Costco Wholesale Corporation', 'Philip Morris International Inc.',
            'Boeing Company', 'Union Pacific Corporation', 'Honeywell International Inc.', 'United Parcel Service Inc.',
            'Caterpillar Inc.', 'General Electric Company',
            'Exxon Mobil Corporation', 'Chevron Corporation', 'ConocoPhillips', 'Schlumberger Limited',
            'NextEra Energy Inc.', 'Duke Energy Corporation', 'Southern Company', 'Dominion Energy Inc.',
            'American Tower Corporation', 'Prologis Inc.', 'Crown Castle International Corp.', 'Public Storage',
            'Linde plc', 'Air Products and Chemicals Inc.', 'Sherwin-Williams Company', 'Newmont Corporation'
        ],
        'GICS Sector': [
            'Information Technology', 'Information Technology', 'Communication Services', 'Communication Services',
            'Information Technology', 'Information Technology', 'Information Technology', 'Information Technology',
            'Information Technology', 'Information Technology',
            'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary',
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
    Returns a dict with DataFrames for income, balance, cashflow, dividends, and history.
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
