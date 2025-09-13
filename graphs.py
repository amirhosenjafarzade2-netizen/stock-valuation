import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
import numpy as np
from datetime import datetime

# Cache data for 1 hour to improve performance
@st.cache_data(ttl=3600)
def cached_fetch_data(ticker, period="10y"):
    """Cached version of fetch_fundamental_data with custom period."""
    stock = yf.Ticker(ticker)
    data = {
        'income': stock.financials.T,
        'balance': stock.balance_sheet.T,
        'cashflow': stock.cashflow.T,
        'dividends': stock.dividends,
        'history': stock.history(period=period)
    }
    return data

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
        st.warning(f"No valid data for {title}. Check if {y_col} is available. Data sample: {df.head() if not df.empty else 'Empty'}")
        return None
    df = df.sort_index().dropna(subset=[y_col])
    if df.empty:
        st.warning(f"No non-NaN data for {title} after cleaning. Data sample: {df.head() if not df.empty else 'Empty'}")
        return None
    
    # Convert index to datetime if x_col is 'Date'
    if x_col == 'Date':
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(subset=[y_col])  # Drop rows where index is NaT
    
    fig = px.line(
        df,
        x=df.index if x_col == 'Date' else df[x_col],
        y=y_col,
        title=title,
        labels={y_col: y_label, x_col: 'Date' if x_col == 'Date' else x_col},
        color_discrete_sequence=['#636EFA'],
        template='plotly_white'
    )
    
    # Customization options
    color = st.color_picker(f"Line Color for {title}", "#636EFA")
    style = st.selectbox(f"Line Style for {title}", ["solid", "dash", "dot"], index=0, key=f"style_{title}")
    fig.update_traces(line_color=color, line_dash=style, hovertemplate=f'<b>%{{y:.2f}}</b><extra></extra>')
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

def calculate_intrinsic_value(cash_flow, growth_rate, discount_rate, years=5):
    """Calculate intrinsic value using DCF."""
    future_cash_flows = [cash_flow * (1 + growth_rate) ** i for i in range(1, years + 1)]
    discounted_flows = [cf / (1 + discount_rate) ** i for i, cf in enumerate(future_cash_flows, 1)]
    return sum(discounted_flows) / (1 + discount_rate)  # Simplified terminal value

def validate_fundamental_data(data):
    """Validate fetched data and provide warnings."""
    income, balance, cashflow, dividends, history = (
        data.get('income', pd.DataFrame()),
        data.get('balance', pd.DataFrame()),
        data.get('cashflow', pd.DataFrame()),
        data.get('dividends', pd.Series()),
        data.get('history', pd.DataFrame())
    )
    issues = []
    if income.empty or income.isna().all().all(): issues.append("No income data")
    if balance.empty or balance.isna().all().all(): issues.append("No balance data")
    if cashflow.empty or cashflow.isna().all().all(): issues.append("No cashflow data")
    if dividends.empty or dividends.isna().all(): issues.append("No dividend data")
    if history.empty or history.isna().all().all(): issues.append("No price history")
    if issues:
        st.warning(f"Data issues detected: {', '.join(issues)} for {ticker}. Try a ticker like AAPL or upload custom data.")
    return all(not df.empty for df in [income, balance, cashflow, history]) if not issues else False

def fetch_alternative_data(ticker, api_key=None):
    """Fetch data from Alpha Vantage as a fallback (requires API key)."""
    if not api_key:
        return None
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        dates = list(data['Time Series (Daily)'].keys())
        prices = [float(data['Time Series (Daily)'][d]['4. close']) for d in dates]
        return pd.DataFrame({'Date': dates, 'Close': prices}).set_index('Date')
    return None

def display_fundamental_graphs(ticker):
    """
    Display fundamental graphs using data from Yahoo Finance.
    """
    st.subheader(f"Fundamental Graphs for {ticker} (as of {datetime.now().strftime('%Y-%m-%d %H:%M %z')})")
    
    # UI Setup
    st.sidebar.title("Graph Options")
    graph_options = ['Revenue', 'EPS', 'ROI', 'Free Cash Flow', 'Expenses', 'Dividends', 'Price', 'P/E Ratio', 'Intrinsic Value']
    selected_graphs = st.sidebar.multiselect("Select Graphs to Display", graph_options, default=graph_options)
    period = st.sidebar.selectbox("Select Time Period", ["1y", "5y", "10y", "max"], index=2)
    show_log = st.sidebar.checkbox("Show Debug Log")
    if st.sidebar.button("Report Issue"):
        st.sidebar.markdown("[Submit Feedback](mailto:support@example.com?subject=Issue%20with%20Stock%20Analyzer)")

    # Fetch and validate data
    data = cached_fetch_data(ticker, period)
    if not validate_fundamental_data(data):
        api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
        if api_key:
            alt_history = fetch_alternative_data(ticker, api_key)
            if alt_history is not None:
                data['history'] = alt_history
                st.success("Fallback to Alpha Vantage data successful.")
        else:
            st.warning("No API key for alternative data. Consider setting ALPHA_VANTAGE_KEY in Streamlit secrets.")
    
    income = data.get('income', pd.DataFrame())
    balance = data.get('balance', pd.DataFrame())
    cashflow = data.get('cashflow', pd.DataFrame())
    dividends = data.get('dividends', pd.Series())
    history = data.get('history', pd.DataFrame())
    
    # Fallback to annual data if quarterly is empty
    try:
        for df in [income, balance, cashflow]:
            if not df.empty:
                df.index = pd.to_datetime(df.index, errors='coerce')
    except Exception as e:
        st.warning(f"Error processing data for {ticker}: {str(e)}")
    
    # Graph rendering with lazy loading and downloads
    if 'Revenue' in selected_graphs:
        revenue_col = 'Total Revenue'
        if 'Total Revenue' not in income.columns:
            possible_revenue_cols = [col for col in income.columns if 'Revenue' in col]
            revenue_col = possible_revenue_cols[0] if possible_revenue_cols else None
        if revenue_col:
            st.write(f"Revenue data sample: {income[revenue_col].head()}")
            revenue_fig = plot_enhanced_line_chart(income, revenue_col, 'Revenue Over Time', 'Revenue ($M)', cagr_periods=len(income))
            if revenue_fig:
                st.plotly_chart(revenue_fig, use_container_width=True)
                st.download_button(label="Download Revenue Data", data=income.to_csv(index=True), file_name="revenue_data.csv", mime="text/csv")
    
    if 'EPS' in selected_graphs:
        eps_col = 'Basic EPS' if 'Basic EPS' in income.columns else ('Diluted EPS' if 'Diluted EPS' in income.columns else None)
        if not eps_col:
            possible_eps_cols = [col for col in income.columns if 'EPS' in col or 'Earnings Per Share' in col]
            eps_col = possible_eps_cols[0] if possible_eps_cols else None
        if eps_col:
            st.write(f"EPS data sample: {income[eps_col].head()}")
            eps_fig = plot_enhanced_line_chart(income, eps_col, f'EPS ({eps_col}) Over Time', 'EPS ($)', cagr_periods=len(income))
            if eps_fig:
                st.plotly_chart(eps_fig, use_container_width=True)
                st.download_button(label="Download EPS Data", data=income.to_csv(index=True), file_name="eps_data.csv", mime="text/csv")
    
    if 'ROI' in selected_graphs:
        if 'Net Income' in income.columns and 'Total Stockholder Equity' in balance.columns:
            common_dates = income.index.intersection(balance.index)
            if not common_dates.empty:
                roe_df = pd.DataFrame(index=common_dates)
                roe_df['Net Income'] = income.loc[common_dates, 'Net Income']
                roe_df['Equity'] = balance.loc[common_dates, 'Total Stockholder Equity']
                roe_df['ROI (ROE)'] = (roe_df['Net Income'] / roe_df['Equity'] * 100).fillna(0).replace([float('inf'), -float('inf')], 0)
                st.write(f"ROI data sample: {roe_df['ROI (ROE)'].head()}")
                roi_fig = plot_enhanced_line_chart(roe_df, 'ROI (ROE)', 'Return on Investment (ROE) Over Time', 'ROI (%)', cagr_periods=len(roe_df))
                if roi_fig:
                    st.plotly_chart(roi_fig, use_container_width=True)
                    st.download_button(label="Download ROI Data", data=roe_df.to_csv(index=True), file_name="roi_data.csv", mime="text/csv")
    
    if 'Free Cash Flow' in selected_graphs:
        fcf_col = 'Free Cash Flow' if 'Free Cash Flow' in cashflow.columns else None
        if not fcf_col:
            possible_fcf_cols = [col for col in cashflow.columns if 'Free Cash Flow' in col]
            fcf_col = possible_fcf_cols[0] if possible_fcf_cols else None
        if fcf_col:
            st.write(f"FCF data sample: {cashflow[fcf_col].head()}")
            fcf_fig = plot_enhanced_line_chart(cashflow, fcf_col, 'Free Cash Flow Over Time', 'FCF ($M)', cagr_periods=len(cashflow))
            if fcf_fig:
                st.plotly_chart(fcf_fig, use_container_width=True)
                st.download_button(label="Download FCF Data", data=cashflow.to_csv(index=True), file_name="fcf_data.csv", mime="text/csv")
    
    if 'Expenses' in selected_graphs:
        exp_col = 'Operating Expense' if 'Operating Expense' in income.columns else ('Cost Of Revenue' if 'Cost Of Revenue' in income.columns else None)
        if not exp_col:
            possible_exp_cols = [col for col in income.columns if 'Expense' in col or 'Cost' in col]
            exp_col = possible_exp_cols[0] if possible_exp_cols else None
        if exp_col:
            st.write(f"Expenses data sample: {income[exp_col].head()}")
            exp_fig = plot_enhanced_line_chart(income, exp_col, f'{exp_col} Over Time', 'Expenses ($M)', cagr_periods=len(income))
            if exp_fig:
                st.plotly_chart(exp_fig, use_container_width=True)
                st.download_button(label="Download Expenses Data", data=income.to_csv(index=True), file_name="expenses_data.csv", mime="text/csv")
    
    if 'Dividends' in selected_graphs:
        if not dividends.empty:
            st.write(f"Raw dividends data: {dividends.head()}")
            dividends_df = dividends.reset_index()
            dividends_df.columns = ['Date', 'Dividend']
            dividends_df['Date'] = pd.to_datetime(dividends_df['Date'], errors='coerce')
            agg_type = st.radio("Dividend Aggregation", ['annual', 'quarterly'], key="div_agg")
            if not dividends_df['Date'].isna().all():
                if agg_type == 'quarterly' and not dividends_df.empty:
                    dividends_df['Quarter'] = dividends_df['Date'].dt.to_period('Q')
                    dividends_quarterly = dividends_df.groupby('Quarter')['Dividend'].sum().reset_index()
                    dividends_quarterly['Date'] = dividends_quarterly['Quarter'].apply(lambda x: x.to_timestamp())
                    df = dividends_quarterly
                else:
                    dividends_annual = dividends_df.groupby(dividends_df['Date'].dt.year)['Dividend'].mean().reset_index()
                    dividends_annual['Date'] = pd.to_datetime(dividends_annual['Date'].astype(str) + '-01-01')
                    df = dividends_annual
                if len(df) >= 1:
                    st.write(f"Aggregated dividends: {df}")
                    div_fig = plot_enhanced_line_chart(
                        df, 'Dividend', 'Dividend Per Share Over Time', 'Dividend ($)',
                        cagr_periods=len(df), x_col='Date'
                    )
                    if div_fig:
                        div_fig.update_traces(hovertemplate='<b>%{y:.2f}</b><extra>Payments: %{customdata}</extra>',
                                            customdata=[len(dividends_df[dividends_df['Date'].dt.year == y]) if 'year' in df.columns else 1 for y in df['Date'].dt.year])
                        st.plotly_chart(div_fig, use_container_width=True)
                        st.download_button(label="Download Dividends Data", data=df.to_csv(index=True), file_name="dividends_data.csv", mime="text/csv")
                    else:
                        st.warning(f"Failed to plot Dividend Per Share graph for {ticker} despite data.")
                else:
                    st.warning(f"No aggregated dividend data points for {ticker} after processing.")
            else:
                st.warning(f"No valid dividend dates for {ticker}.")
        else:
            st.warning(f"No dividend data available for {ticker}.")
    
    if 'Price' in selected_graphs:
        if not history.empty:
            st.write(f"Price data sample: {history['Close'].head()}")
            price_fig = plot_enhanced_line_chart(
                history, 'Close', 'Stock Price Over Time', 'Price ($)', cagr_periods=len(history) / 252
            )
            if price_fig:
                st.plotly_chart(price_fig, use_container_width=True)
                st.download_button(label="Download Price Data", data=history.to_csv(index=True), file_name="price_data.csv", mime="text/csv")
    
    if 'P/E Ratio' in selected_graphs:
        if not history.empty and eps_col:
            latest_eps = income[eps_col].dropna().iloc[-1] if not income[eps_col].dropna().empty else 1.0
            history_copy = history.copy()
            history_copy['PE Ratio'] = history_copy['Close'] / latest_eps
            st.write(f"P/E data sample: {history_copy['PE Ratio'].head()}")
            pe_fig = plot_enhanced_line_chart(history_copy, 'PE Ratio', 'P/E Ratio Over Time', 'P/E Ratio')
            if pe_fig:
                st.plotly_chart(pe_fig, use_container_width=True)
                st.download_button(label="Download P/E Data", data=history_copy.to_csv(index=True), file_name="pe_data.csv", mime="text/csv")
    
    if 'Intrinsic Value' in selected_graphs:
        with st.expander("Intrinsic Value Comparison"):
            st.info("Adjust parameters to calculate intrinsic value. Data source: User input with DCF approximation.")
            cash_flow = st.number_input("Annual Cash Flow ($M)", value=10.0, min_value=0.0)
            growth_rate = st.slider("Growth Rate (%)", 0.0, 10.0, 2.0) / 100
            discount_rate = st.slider("Discount Rate (%)", 0.0, 15.0, 5.0) / 100
            intrinsic_value = calculate_intrinsic_value(cash_flow * 1e6, growth_rate, discount_rate)
            st.write(f"Calculated Intrinsic Value: ${intrinsic_value:,.2f}")
            if not history.empty:
                latest_date = history.index[-1]
                latest_price = history['Close'].iloc[-1]
                if pd.notna(latest_price) and latest_price > 0:
                    val_df = pd.DataFrame({
                        'Date': [latest_date, latest_date],
                        'Value': [intrinsic_value, latest_price],
                        'Metric': ['Intrinsic Value', 'Price']
                    })
                    st.write(f"Intrinsic Value data: {val_df}")
                    val_fig = px.line(
                        val_df, x='Date', y='Value', color='Metric', title='Intrinsic Value vs. Price',
                        labels={'Value': 'Value ($)', 'Metric': 'Metric'},
                        color_discrete_map={'Intrinsic Value': '#00CC96', 'Price': '#EF553B'}
                    )
                    val_fig.update_layout(
                        template='plotly_white', height=450, hovermode='x unified', showlegend=True
                    )
                    val_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    val_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    st.plotly_chart(val_fig, use_container_width=True)
                    st.download_button(label="Download Intrinsic Value Data", data=val_df.to_csv(index=False), file_name="intrinsic_value_data.csv", mime="text/csv")
                else:
                    st.warning(f"No valid price data for Intrinsic Value Comparison for {ticker}.")
            else:
                st.warning(f"No price history available for {ticker}.")
    
    # Debug log
    if show_log:
        st.sidebar.text_area("Log", value=f"Income columns: {income.columns}\nHistory dates: {history.index}", height=200)

if __name__ == "__main__":
    ticker = st.text_input("Enter Ticker", "AAPL")
    if ticker:
        display_fundamental_graphs(ticker)
