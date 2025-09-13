import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from data_fetch import fetch_fundamental_data  # Import to fetch cached data

def plot_line_chart(df, y_col, title, y_label):
    """
    Helper to plot a line chart with Plotly.
    """
    if df.empty or y_col not in df.columns or df[y_col].isna().all():
        return None
    df = df.sort_index()  # Ensure chronological order
    fig = px.line(df, x=df.index, y=y_col, title=title)
    fig.update_layout(
        yaxis_title=y_label,
        xaxis_title="Date",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )
    return fig

def display_fundamental_graphs(ticker):
    """
    Display Qualtrim-inspired fundamental graphs for the ticker.
    """
    st.subheader(f"Fundamental Graphs for {ticker}")
    
    # Fetch fundamental data
    data = fetch_fundamental_data(ticker)
    income = data.get('income', pd.DataFrame())
    balance = data.get('balance', pd.DataFrame())
    cashflow = data.get('cashflow', pd.DataFrame())
    dividends = data.get('dividends', pd.Series())
    history = data.get('history', pd.DataFrame())
    
    # Revenue Over Time
    if 'Total Revenue' in income.columns:
        revenue_fig = plot_line_chart(income, 'Total Revenue', 'Revenue Over Time', 'Revenue ($)')
        if revenue_fig:
            st.plotly_chart(revenue_fig, use_container_width=True)
        else:
            st.info("Revenue data not available.")
    
    # EPS Growth
    if 'Eps Diluted' in income.columns:
        eps_fig = plot_line_chart(income, 'Eps Diluted', 'EPS (Diluted) Over Time', 'EPS ($)')
        if eps_fig:
            st.plotly_chart(eps_fig, use_container_width=True)
        else:
            st.info("EPS data not available.")
    
    # Free Cash Flow
    if 'Free Cash Flow' in cashflow.columns:
        fcf_fig = plot_line_chart(cashflow, 'Free Cash Flow', 'Free Cash Flow Over Time', 'FCF ($)')
        if fcf_fig:
            st.plotly_chart(fcf_fig, use_container_width=True)
        else:
            st.info("Free Cash Flow data not available.")
    
    # Dividend History
    if not dividends.empty:
        dividends_df = dividends.reset_index()
        dividends_df.columns = ['Date', 'Dividend']
        dividends_fig = plot_line_chart(dividends_df, 'Dividend', 'Dividend History', 'Dividend Per Share ($)')
        if dividends_fig:
            st.plotly_chart(dividends_fig, use_container_width=True)
        else:
            st.info("Dividend data not available.")
    
    # Gross Margin
    if 'Gross Profit' in income.columns and 'Total Revenue' in income.columns:
        income['Gross Margin'] = (income['Gross Profit'] / income['Total Revenue'] * 100).replace([float('inf'), -float('inf')], 0)
        margin_fig = plot_line_chart(income, 'Gross Margin', 'Gross Margin Over Time', 'Gross Margin (%)')
        if margin_fig:
            st.plotly_chart(margin_fig, use_container_width=True)
        else:
            st.info("Gross Margin data not available.")
    
    # Debt Levels
    if 'Total Debt' in balance.columns:
        debt_fig = plot_line_chart(balance, 'Total Debt', 'Total Debt Over Time', 'Debt ($)')
        if debt_fig:
            st.plotly_chart(debt_fig, use_container_width=True)
        else:
            st.info("Debt data not available.")
    
    # Return on Equity
    if 'Net Income' in income.columns and 'Total Stockholder Equity' in balance.columns:
        balance['ROE'] = (income['Net Income'] / balance['Total Stockholder Equity'] * 100).replace([float('inf'), -float('inf')], 0)
        roe_fig = plot_line_chart(balance, 'ROE', 'Return on Equity Over Time', 'ROE (%)')
        if roe_fig:
            st.plotly_chart(roe_fig, use_container_width=True)
        else:
            st.info("ROE data not available.")
    
    # P/E Ratio History
    if not history.empty and 'Eps Diluted' in income.columns:
        history['Date'] = history.index
        merged = history.merge(income[['Eps Diluted']], left_index=True, right_index=True, how='left')
        merged['Eps Diluted'] = merged['Eps Diluted'].ffill()
        merged['PE Ratio'] = merged['Close'] / merged['Eps Diluted'].replace(0, float('nan'))
        pe_fig = plot_line_chart(merged, 'PE Ratio', 'P/E Ratio Over Time', 'P/E Ratio')
        if pe_fig:
            st.plotly_chart(pe_fig, use_container_width=True)
        else:
            st.info("P/E Ratio data not available.")
    
    # Placeholder for unavailable graphs
    st.info("Note: Some Qualtrim graphs (e.g., Revenue by Segment, Custom KPIs) are not available due to yfinance limitations. Consider a premium API for more data.")

if __name__ == "__main__":
    ticker = st.text_input("Enter Ticker", "AAPL")
    if ticker:
        display_fundamental_graphs(ticker)
