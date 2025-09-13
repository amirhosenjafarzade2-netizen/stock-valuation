import streamlit as st
import pandas as pd
from data_fetch import fetch_stock_data, get_sp500_tickers
from valuation_models import calculate_valuation
from utils import validate_inputs

def run_screener(model, min_undervaluation=0.0, selected_sectors=None):
    """
    Screen S&P 500 stocks using the specified valuation model.
    Returns a DataFrame with undervalued stocks, including sector, ticker, name, undervaluation %, and market cap.
    """
    # Fetch S&P 500 tickers and metadata
    sp500_data = get_sp500_tickers()
    
    # Filter by selected sectors if provided
    if selected_sectors:
        sp500_data = sp500_data[sp500_data['GICS Sector'].isin(selected_sectors)]
    
    results = []
    progress_bar = st.progress(0)
    total_stocks = len(sp500_data)
    
    for idx, row in sp500_data.iterrows():
        ticker = row['Symbol']
        try:
            # Fetch stock data
            data = fetch_stock_data(ticker)
            data['model'] = model  # Set the valuation model
            
            # Validate inputs
            if validate_inputs(data):
                # Calculate valuation
                valuation_results = calculate_valuation(data)
                undervaluation = valuation_results.get('undervaluation', 0)
                
                # Only include undervalued stocks above threshold
                if undervaluation > min_undervaluation:
                    results.append({
                        'Sector': row['GICS Sector'],
                        'Ticker': ticker,
                        'Name': row['Security'],
                        'Undervaluation %': undervaluation,
                        'Market Cap (B)': data.get('market_cap', 0) / 1e9,  # Convert to billions
                        'Intrinsic Value': valuation_results.get('intrinsic_value', 0)
                    })
            
            # Update progress
            progress_bar.progress((idx + 1) / total_stocks)
            
        except Exception as e:
            st.warning(f"Skipping {ticker}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by market cap (descending) by default
    if not results_df.empty:
        results_df = results_df.sort_values(by='Market Cap (B)', ascending=False)
    
    return results_df

def display_screener():
    """
    Display the stock screener interface in Streamlit.
    """
    st.header("S&P 500 Stock Screener")
    st.markdown("Screen S&P 500 stocks for undervaluation based on the selected valuation model.")
    
    # Get S&P 500 data for sector options
    sp500_data = get_sp500_tickers()
    sector_options = sorted(sp500_data['GICS Sector'].unique())
    
    # User inputs
    model = st.selectbox(
        "Valuation Model",
        ["Core Valuation (Excel)", "Lynch Method", "Discounted Cash Flow (DCF)", 
         "Dividend Discount Model (DDM)", "Two-Stage DCF", "Residual Income (RI)", 
         "Reverse DCF", "Graham Intrinsic Value"],
        key="screener_model"
    )
    
    min_undervaluation = st.number_input(
        "Minimum Undervaluation %",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        help="Show only stocks with undervaluation above this threshold."
    )
    
    selected_sectors = st.multiselect(
        "Filter by Sector(s)",
        sector_options,
        default=sector_options,
        help="Select sectors to screen. Leave all selected for full S&P 500."
    )
    
    sort_by = st.selectbox(
        "Sort By",
        ["Market Cap (Descending)", "Undervaluation % (Descending)"],
        help="Sort results by market cap or undervaluation percentage."
    )
    
    if st.button("Run Screener"):
        with st.spinner("Screening S&P 500 stocks... This may take a few minutes."):
            results_df = run_screener(model, min_undervaluation, selected_sectors)
            
            if results_df.empty:
                st.info("No undervalued stocks found with the given criteria.")
            else:
                # Apply sorting
                if sort_by == "Undervaluation % (Descending)":
                    results_df = results_df.sort_values(by='Undervaluation %', ascending=False)
                # Else already sorted by Market Cap (B) in run_screener
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Screener Results",
                    data=csv,
                    file_name="sp500_screener_results.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    display_screener()
