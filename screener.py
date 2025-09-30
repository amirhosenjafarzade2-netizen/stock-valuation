import streamlit as st
import pandas as pd
import asyncio
import aiohttp
from data_fetch import fetch_stock_data, get_sp500_tickers
from valuation_models import calculate_valuation
from utils import validate_inputs

async def fetch_stock_data_async(ticker, session):
    """Async wrapper for fetch_stock_data."""
    try:
        return ticker, await asyncio.to_thread(fetch_stock_data, ticker)
    except Exception as e:
        st.warning(f"Skipping {ticker}: {str(e)}")
        return ticker, None

async def run_screener_async(model, min_undervaluation=0.0, selected_sectors=None):
    """
    Screen S&P 500 stocks asynchronously using the specified valuation model.
    """
    sp500_data = get_sp500_tickers()
    
    if selected_sectors:
        sp500_data = sp500_data[sp500_data['GICS Sector'].isin(selected_sectors)]
    
    results = []
    progress_bar = st.progress(0)
    total_stocks = len(sp500_data)
    batch_size = 10  # NEW: Process in batches
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, total_stocks, batch_size):
            batch = sp500_data.iloc[i:i + batch_size]
            tasks = [fetch_stock_data_async(row['Symbol'], session) for _, row in batch.iterrows()]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for idx, (ticker, data) in enumerate(batch_results):
                if data is None:
                    continue
                try:
                    data['model'] = model
                    if validate_inputs(data):
                        valuation_results = calculate_valuation(data)
                        undervaluation = valuation_results.get('undervaluation', None)
                        
                        if undervaluation is not None and undervaluation > min_undervaluation:
                            row = sp500_data[sp500_data['Symbol'] == ticker].iloc[0]
                            results.append({
                                'Sector': row['GICS Sector'],
                                'Ticker': ticker,
                                'Name': row['Security'],
                                'Undervaluation %': undervaluation,
                                'Market Cap (B)': data.get('market_cap', 0) / 1e9 if data.get('market_cap') else None,
                                'Intrinsic Value': valuation_results.get('intrinsic_value', None)
                            })
                except Exception as e:
                    st.warning(f"Error processing {ticker}: {str(e)}")
                progress_bar.progress(min((i + idx + 1) / total_stocks, 1.0))
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Market Cap (B)', ascending=False, na_position='last')
    
    return results_df

def display_screener():
    """
    Display the stock screener interface in Streamlit.
    """
    st.header("S&P 500 Stock Screener")
    st.markdown("Screen S&P 500 stocks for undervaluation based on the selected valuation model.")
    
    sp500_data = get_sp500_tickers()
    sector_options = sorted(sp500_data['GICS Sector'].unique())
    
    model = st.selectbox(
        "Valuation Model",
        ["Core Valuation (Excel)", "Lynch Method", "Discounted Cash Flow (DCF)", 
         "Dividend Discount Model (DDM)", "Two-Stage DCF", "Residual Income (RI)", 
         "Reverse DCF", "Graham Intrinsic Value"],
        key="screener_model"
    )
    
    min_undervaluation = st.number_input(
        "Minimum Undervaluation %",
        min_value=0.0, max_value=100.0, value=0.0,
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
        with st.spinner("Screening S&P 500 stocks..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results_df = loop.run_until_complete(run_screener_async(model, min_undervaluation, selected_sectors))
            loop.close()
            
            if results_df.empty:
                st.info("No undervalued stocks found with the given criteria.")
            else:
                if sort_by == "Undervaluation % (Descending)":
                    results_df = results_df.sort_values(by='Undervaluation %', ascending=False, na_position='last')
                st.dataframe(results_df, use_container_width=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Screener Results",
                    data=csv,
                    file_name="sp500_screener_results.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    display_screener()
