import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from data_fetch import fetch_fundamental_data
from datetime import datetime

def calculate_cagr(start_value, end_value, periods):
    """Calculate CAGR, handling negative start values."""
    if periods <= 0 or start_value == 0:
        return None
    if start_value < 0 or end_value < 0:
        return None  # CHANGED: Skip negative values
    return ((end_value / start_value) ** (1 / periods) - 1) * 100

def plot_enhanced_line_chart(df, y_col, title, y_label, x_col='Date', cagr_periods=None):
    """
    Enhanced line chart with interactivity, annotations, and CAGR.
    """
    if df.empty or y_col not in df.columns or df[y_col].isna().all():
        st.warning(f"No valid data for {title}. Check if {y_col} is available.")
        return None
    df = df.sort_index().dropna(subset=[y_col])
    if df.empty:
        st.warning(f"No non-NaN data for {title} after cleaning.")
        return None
    
    if x_col == 'Date':
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(subset=[y_col])
    
    fig = px.line(
        df,
        x=df.index if x_col == 'Date' else df[x_col],
        y=y_col,
        title=title,
        labels={y_col: y_label, x_col: 'Date' if x_col == 'Date' else x_col},
        color_discrete_sequence=['#636EFA'],
        template='plotly_white'
    )
    
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
        if cagr is not None:
            fig.add_annotation(
                x=df.index[-1],
                y=end_val,
                text=f"CAGR: {cagr:.1f}%",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                font=dict(size=12),
                xanchor="left",
                yanchor="top"
            )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def display_fundamental_graphs(ticker):
    """
    Display fundamental graphs using data from Yahoo Finance.
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
                df.index = pd.to_datetime(df.index, errors='coerce')
    except Exception as e:
        st.warning(f"Error fetching annual data for {ticker}: {str(e)}")
    
    # Revenue Over Time
    revenue_col = None
    if 'Total Revenue' in income.columns:
        revenue_col = 'Total Revenue'
    else:
        possible_revenue_cols = [col for col in income.columns if 'Total Revenue' in col or 'Revenue' in col]
        revenue_col = possible_revenue_cols[0] if possible_revenue_cols else None
    if revenue_col:
        periods = len(income) / 4 if 'quarterly' in str(income.index) else len(income)  # CHANGED: Adjust for quarterly
        revenue_fig = plot_enhanced_line_chart(
            income, revenue_col, 'Revenue Over Time', 'Revenue ($M)', cagr_periods=periods
        )
        if revenue_fig:
            st.plotly_chart(revenue_fig, use_container_width=True)
    
    # EPS
    eps_col = None
    for col in ['Basic EPS', 'Diluted EPS']:
        if col in income.columns:
            eps_col = col
            break
    if not eps_col:
        possible_eps_cols = [col for col in income.columns if 'EPS' in col or 'Earnings Per Share' in col]
        eps_col = possible_eps_cols[0] if possible_eps_cols else None
    if eps_col:
        periods = len(income) / 4 if 'quarterly' in str(income.index) else len(income)
        eps_fig = plot_enhanced_line_chart(
            income, eps_col, f'EPS ({eps_col}) Over Time', 'EPS ($)', cagr_periods=periods
        )
        if eps_fig:
            st.plotly_chart(eps_fig, use_container_width=True)
    
    # ROE (not ROI)
    if 'Net Income' in income.columns and 'Total Stockholder Equity' in balance.columns:
        common_dates = income.index.intersection(balance.index)
        if not common_dates.empty:
            roe_df = pd.DataFrame(index=common_dates)
            roe_df['ROE'] = (income.loc[common_dates, 'Net Income'] / 
                            balance.loc[common_dates, 'Total Stockholder Equity'] * 100)
            periods = len(roe_df) / 4 if 'quarterly' in str(roe_df.index) else len(roe_df)
            roe_fig = plot_enhanced_line_chart(
                roe_df, 'ROE', 'Return on Equity (ROE) Over Time', 'ROE (%)', cagr_periods=periods
            )
            if roe_fig:
                st.plotly_chart(roe_fig, use_container_width=True)
    
    # Free Cash Flow
    fcf_col = None
    if 'Free Cash Flow' in cashflow.columns:
        fcf_col = 'Free Cash Flow'
    else:
        possible_fcf_cols = [col for col in cashflow.columns if 'Free Cash Flow' in col]
        fcf_col = possible_fcf_cols[0] if possible_fcf_cols else None
    if fcf_col:
        periods = len(cashflow) / 4 if 'quarterly' in str(cashflow.index) else len(cashflow)
        fcf_fig = plot_enhanced_line_chart(
            cashflow, fcf_col, 'Free Cash Flow Over Time', 'FCF ($M)', cagr_periods=periods
        )
        if fcf_fig:
            st.plotly_chart(fcf_fig, use_container_width=True)
    
    # Expenses
    exp_col = None
    for col in ['Operating Expense', 'Cost Of Revenue']:
        if col in income.columns:
            exp_col = col
            break
    if not exp_col:
        possible_exp_cols = [col for col in income.columns if 'Expense' in col or 'Cost' in col]
        exp_col = possible_exp_cols[0] if possible_exp_cols else None
    if exp_col:
        periods = len(income) / 4 if 'quarterly' in str(income.index) else len(income)
        exp_fig = plot_enhanced_line_chart(
            income, exp_col, f'{exp_col} Over Time', 'Expenses ($M)', cagr_periods=periods
        )
        if exp_fig:
            st.plotly_chart(exp_fig, use_container_width=True)
    
    # Dividends
    if not dividends.empty:
        dividends_df = dividends.reset_index()
        dividends_df.columns = ['Date', 'Dividend']
        dividends_df['Date'] = pd.to_datetime(dividends_df['Date'], errors='coerce')
        if not dividends_df['Date'].isna().all():
            dividends_annual = dividends_df.groupby(dividends_df['Date'].dt.year)['Dividend'].sum().reset_index()
            dividends_annual['Date'] = pd.to_datetime(dividends_annual['Date'].astype(str) + '-01-01')
            if len(dividends_annual) >= 2:
                div_fig = plot_enhanced_line_chart(
                    dividends_annual,
                    'Dividend',
                    'Dividend Per Share Over Time',
                    'Dividend ($)',
                    cagr_periods=len(dividends_annual)
                )
                if div_fig:
                    st.plotly_chart(div_fig, use_container_width=True)
            else:
                st.warning(f"Insufficient dividend data points for {ticker}.")
        else:
            st.warning(f"No valid dividend dates for {ticker}.")
    else:
        st.warning(f"No dividend data available for {ticker}.")
    
    # Price
    if not history.empty:
        periods = len(history) / 252  # Trading days per year
        price_fig = plot_enhanced_line_chart(
            history, 'Close', 'Stock Price Over Time', 'Price ($)', cagr_periods=periods
        )
        if price_fig:
            st.plotly_chart(price_fig, use_container_width=True)
    
    # P/E Ratio
    if not history.empty and eps_col:
        latest_eps = income[eps_col].dropna().iloc[-1] if not income[eps_col].dropna().empty else None
        if latest_eps:
            history_copy = history.copy()
            history_copy['PE Ratio'] = history_copy['Close'] / latest_eps
            pe_fig = plot_enhanced_line_chart(
                history_copy, 'PE Ratio', 'P/E Ratio Over Time', 'P/E Ratio'
            )
            if pe_fig:
                st.plotly_chart(pe_fig, use_container_width=True)
    
    # Intrinsic Value Comparison
    with st.expander("Intrinsic Value Comparison"):
        st.info("Enter an intrinsic value to compare with the current stock price.")
        intr_val = st.number_input("Enter Intrinsic Value ($)", value=100.0, min_value=0.0)
        if not history.empty:
            latest_date = history.index[-1]
            latest_price = history['Close'].iloc[-1]
            val_df = pd.DataFrame({
                'Date': [latest_date, latest_date],
                'Value': [intr_val, latest_price],
                'Metric': ['Intrinsic Value', 'Price']
            })
            val_fig = px.line(
                val_df,
                x='Date',
                y='Value',
                color='Metric',
                title='Intrinsic Value vs. Price',
                labels={'Value': 'Value ($)', 'Metric': 'Metric'},
                color_discrete_map={'Intrinsic Value': '#00CC96', 'Price': '#EF553B'}
            )
            val_fig.update_layout(
                template='plotly_white',
                height=450,
                hovermode='x unified',
                showlegend=True
            )
            val_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            val_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            st.plotly_chart(val_fig, use_container_width=True)
        else:
            st.warning(f"No price history available for {ticker}.")
