import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from data_fetch import fetch_fundamental_data
from datetime import datetime

def calculate_cagr(start_value, end_value, periods):
    """Calculate CAGR for annotations."""
    if periods > 0 and start_value > 0:
        return ((end_value / start_value) ** (1 / periods) - 1) * 100
    return 0

def plot_enhanced_line_chart(df, y_col, title, y_label, x_col='Date', cagr_periods=None):
    """
    Enhanced line chart with interactivity, annotations, and CAGR.
    """
    if df.empty or y_col not in df.columns or df[y_col].isna().all():
        return None
    df = df.sort_index().dropna(subset=[y_col])
    if df.empty:
        return None
    
    fig = px.line(df, x=df.index if x_col == 'Date' else df[x_col], y=y_col, title=title,
                  labels={y_col: y_label, x_col: 'Date' if x_col == 'Date' else x_col},
                  color_discrete_sequence=['#636EFA'],
                  template='plotly_white')
    
    fig.update_traces(hovertemplate=f'<b>%{{y:.2f}}</b><extra></extra>')
    fig.update_layout(
        yaxis_title=y_label,
        xaxis_title='Date',
        template='plotly_white',
        margin=dict(l=60, r=60, t=80, b=60),
        height=450,
        hovermode='x unified',
        showlegend=False
    )
    
    if cagr_periods and len(df) >= 2:
        start_val = df[y_col].iloc[0]
        end_val = df[y_col].iloc[-1]
        cagr = calculate_cagr(start_val, end_val, cagr_periods)
        fig.add_annotation(x=df.index[-1], y=end_val,
                           text=f"CAGR: {cagr:.1f}%",
                           showarrow=False, bgcolor="white", bordercolor="black",
                           font=dict(size=12), xanchor="left", yanchor="top")
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def plot_revenue_by_segment(years, segment_data, title="Revenue by Product Segment"):
    """
    Plot revenue share by product segment with separate colors.
    """
    if not segment_data or not years:
        return None
    
    df = pd.DataFrame(index=years)
    for seg_name, revenues in segment_data.items():
        df[seg_name] = revenues
    
    df = df.fillna(0).clip(lower=0)
    
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], stackgroup='one', name=col,
                                line=dict(width=0.5), fill='tonexty',
                                hovertemplate='<b>%{y:.2f}</b> ($M)<extra></extra>',
                                marker_color=px.colors.qualitative.Plotly[len(fig.data)]))
    
    fig.update_layout(
        title=title,
        yaxis_title='Revenue ($M)',
        xaxis_title='Year',
        template='plotly_white',
        margin=dict(l=60, r=60, t=80, b=60),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def display_fundamental_graphs(ticker):
    """
    Enhanced Qualtrim-inspired fundamental graphs with all requested metrics.
    """
    st.subheader(f"Fundamental Graphs for {ticker} (as of {datetime.now().strftime('%Y-%m-%d %H:%M %z')})")
    
    data = fetch_fundamental_data(ticker)
    income = data.get('income', pd.DataFrame())
    balance = data.get('balance', pd.DataFrame())
    cashflow = data.get('cashflow', pd.DataFrame())
    dividends = data.get('dividends', pd.Series())
    history = data.get('history', pd.DataFrame())
    
    try:
        stock = yf.Ticker(ticker)
        income_annual = stock.financials.T
        balance_annual = stock.balance_sheet.T
        cashflow_annual = stock.cashflow.T
        income = income_annual if not income_annual.empty else income
        balance = balance_annual if not balance_annual.empty else balance
        cashflow = cashflow_annual if not cashflow_annual.empty else cashflow
        for df in [income, balance, cashflow]:
            if not df.empty:
                df.index = pd.to_datetime(df.index)
    except:
        pass
    
    # Revenue Over Time
    if 'Total Revenue' in income.columns:
        revenue_fig = plot_enhanced_line_chart(income, 'Total Revenue', 'Revenue Over Time', 'Revenue ($M)', cagr_periods=len(income))
        if revenue_fig:
            st.plotly_chart(revenue_fig, use_container_width=True)
    
    # Revenue by Segment
    with st.expander("Revenue by Product Segment"):
        st.info("Enter manual data for segments; Qualtrim uses premium sources for this.")
        years = income.index.year.unique() if not income.empty else [2020, 2021, 2022, 2023, 2024, 2025]
        segments = {}
        for i in range(3):  # Allow up to 3 segments
            seg_name = st.text_input(f"Segment {i+1} Name", f"Product {i+1}" if i < 2 else "")
            if seg_name:
                revenues = [st.number_input(f"{seg_name} Revenue ($M) for {year}", value=0.0) for year in years]
                segments[seg_name] = revenues
        if segments:
            seg_fig = plot_revenue_by_segment(years, segments)
            if seg_fig:
                st.plotly_chart(seg_fig, use_container_width=True)
    
    # Shares Outstanding and Buybacks (approximated)
    if not history.empty:
        shares_out = history['Volume'] / history['Close']  # Rough estimate
        shares_df = pd.DataFrame({'Shares Outstanding': shares_out}, index=history.index)
        buyback_change = shares_df['Shares Outstanding'].pct_change() * -1  # Negative change as buyback proxy
        shares_fig = plot_enhanced_line_chart(shares_df, 'Shares Outstanding', 'Shares Outstanding Over Time', 'Shares (M)', cagr_periods=len(shares_df))
        if shares_fig:
            st.plotly_chart(shares_fig, use_container_width=True)
        buyback_fig = plot_enhanced_line_chart(shares_df.iloc[1:], buyback_change.name, 'Buyback Impact (Approx.)', 'Change %', cagr_periods=len(shares_df)-1)
        if buyback_fig:
            st.plotly_chart(buyback_fig, use_container_width=True)
    
    # Cash and Debt
    if 'Cash' in balance.columns and 'Total Debt' in balance.columns:
        cash_debt_df = pd.DataFrame({
            'Cash': balance['Cash'],
            'Total Debt': balance['Total Debt']
        }, index=balance.index)
        cash_debt_fig = px.line(cash_debt_df, title='Cash and Debt Over Time', labels={'value': 'Amount ($M)', 'variable': 'Metric'},
                               color_discrete_map={'Cash': '#00CC96', 'Total Debt': '#EF553B'})
        cash_debt_fig.update_layout(template='plotly_white', height=450, hovermode='x unified')
        cash_debt_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        cash_debt_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        st.plotly_chart(cash_debt_fig, use_container_width=True)
    
    # EPS
    eps_col = 'BasicEPS' if 'BasicEPS' in income.columns else ('DilutedEPS' if 'DilutedEPS' in income.columns else None)
    if eps_col:
        eps_fig = plot_enhanced_line_chart(income, eps_col, f'EPS ({eps_col[:-3]}) Over Time', 'EPS ($)', cagr_periods=len(income))
        if eps_fig:
            st.plotly_chart(eps_fig, use_container_width=True)
    
    # ROI (Approximated via ROE or ROIC)
    if 'Net Income' in income.columns and 'Total Stockholder Equity' in balance.columns:
        common_dates = income.index.intersection(balance.index)
        roe_df = pd.DataFrame(index=common_dates)
        roe_df['Net Income'] = income.loc[common_dates, 'Net Income']
        roe_df['Equity'] = balance.loc[common_dates, 'Total Stockholder Equity']
        roe_df['ROI (ROE)'] = (roe_df['Net Income'] / roe_df['Equity'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
        roi_fig = plot_enhanced_line_chart(roe_df, 'ROI (ROE)', 'Return on Investment (ROE) Over Time', 'ROI (%)', cagr_periods=len(roe_df))
        if roi_fig:
            st.plotly_chart(roi_fig, use_container_width=True)
    
    # Free Cash Flow
    if 'FreeCashFlow' in cashflow.columns:
        fcf_fig = plot_enhanced_line_chart(cashflow, 'FreeCashFlow', 'Free Cash Flow Over Time', 'FCF ($M)', cagr_periods=len(cashflow))
        if fcf_fig:
            st.plotly_chart(fcf_fig, use_container_width=True)
    
    # Expenses (Operating Expenses as proxy)
    if 'OperatingExpense' in income.columns:
        exp_fig = plot_enhanced_line_chart(income, 'OperatingExpense', 'Operating Expenses Over Time', 'Expenses ($M)', cagr_periods=len(income))
        if exp_fig:
            st.plotly_chart(exp_fig, use_container_width=True)
    elif 'CostOfRevenue' in income.columns:
        exp_fig = plot_enhanced_line_chart(income, 'CostOfRevenue', 'Cost of Revenue Over Time', 'Expenses ($M)', cagr_periods=len(income))
        if exp_fig:
            st.plotly_chart(exp_fig, use_container_width=True)
    
    # Dividends
    if not dividends.empty:
        dividends_df = dividends.reset_index()
        dividends_df.columns = ['Date', 'Dividend']
        div_fig = plot_enhanced_line_chart(dividends_df, 'Dividend', 'Dividend Per Share Over Time', 'Dividend ($)', cagr_periods=len(dividends_df))
        if div_fig:
            st.plotly_chart(div_fig, use_container_width=True)
    
    # Price
    if not history.empty:
        price_fig = plot_enhanced_line_chart(history, 'Close', 'Stock Price Over Time', 'Price ($)', cagr_periods=len(history))
        if price_fig:
            st.plotly_chart(price_fig, use_container_width=True)
    
    # Valuation Graphs (P/E Ratio and Intrinsic Value Placeholder)
    if not history.empty and eps_col:
        latest_eps = income[eps_col].dropna().iloc[-1] if not income[eps_col].dropna().empty else 1.0
        history_copy = history.copy()
        history_copy['PE Ratio'] = history_copy['Close'] / latest_eps
        pe_fig = plot_enhanced_line_chart(history_copy, 'PE Ratio', 'P/E Ratio Over Time', 'P/E Ratio')
        if pe_fig:
            st.plotly_chart(pe_fig, use_container_width=True)
    
    # Intrinsic Value (Placeholder - requires valuation model integration)
    with st.expander("Intrinsic Value Comparison"):
        st.info("Intrinsic value requires valuation model data (e.g., DCF from app.py). Placeholder for now.")
        intr_val = st.number_input("Enter Intrinsic Value ($)", value=100.0)
        val_df = pd.DataFrame({'Date': [history.index[-1]], 'Intrinsic Value': [intr_val], 'Price': [history['Close'].iloc[-1]]})
        val_fig = px.line(val_df, x='Date', y=['Intrinsic Value', 'Price'], title='Intrinsic Value vs. Price',
                         labels={'value': 'Value ($)', 'variable': 'Metric'},
                         color_discrete_map={'Intrinsic Value': '#00CC96', 'Price': '#EF553B'})
        val_fig.update_layout(template='plotly_white', height=450, hovermode='x unified')
        val_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        val_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        st.plotly_chart(val_fig, use_container_width=True)
    
    st.info("""
    **Inspired by Qualtrim**: These charts aim to mimic Qualtrim's depth (e.g., 30+ years, buyback impacts). yfinance limits data to ~10 years. Upload Qualtrim screenshots for further refinement!
    """)

if __name__ == "__main__":
    ticker = st.text_input("Enter Ticker", "AAPL")
    if ticker:
        display_fundamental_graphs(ticker)
