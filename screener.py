import streamlit as st
import pandas as pd
import asyncio
try:
    import aiohttp
except ImportError:
    st.error("The 'aiohttp' library is not installed. Please add 'aiohttp>=3.8.4' to requirements.txt and install it.")
    st.stop()
from data_fetch import fetch_stock_data, get_sp500_tickers
from valuation_models import calculate_valuation
from utils import validate_inputs

async def fetch_stock_data_async(ticker, session):
    """Async wrapper for fetch_stock_data with enhanced error handling."""
    try:
        return ticker, await asyncio.to_thread(fetch_stock_data, ticker)
    except Exception as e:
        st.warning(f"Failed to fetch data for {ticker}: {str(e)}")
        return ticker, None

async def run_screener_async(model, min_undervaluation=0.0, selected_sectors=None, max_stocks=500):
    """
    Screen S&P 500 stocks asynchronously using the specified valuation model.
    Args:
        model (str): Valuation model to use.
        min_undervaluation (float): Minimum undervaluation percentage threshold.
        selected_sectors (list): List of sectors to filter, or None for all.
        max_stocks (int): Maximum number of stocks to process.
    Returns:
        pd.DataFrame: Screener results with ticker, sector, undervaluation, etc.
    """
    try:
        sp500_data = get_sp500_tickers()
    except Exception as e:
        st.error(f"Error fetching S&P 500 tickers: {str(e)}")
        return pd.DataFrame()

    if selected_sectors:
        sp500_data = sp500_data[sp500_data['GICS Sector'].isin(selected_sectors)]

    sp500_data = sp500_data.head(max_stocks)
    results = []
    progress_bar = st.progress(0)
    total_stocks = len(sp500_data)
    batch_size = 10

    async with aiohttp.ClientSession() as session:
        for i in range(0, total_stocks, batch_size):
            batch = sp500_data.iloc[i:i + batch_size]
            tasks = [fetch_stock_data_async(row['Symbol'], session) for _, row in batch.iterrows()]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, (ticker, data) in enumerate(batch_results):
                if data is None or 'error' in data:
                    continue
                try:
                    data['model'] = model
                    if validate_inputs(data):
                        valuation_results = calculate_valuation(data)
                        undervaluation = valuation_results.get('undervaluation')
                        if undervaluation is not None and undervaluation > min_undervaluation:
                            row = sp500_data[sp500_data['Symbol'] == ticker].iloc[0]
                            results.append({
                                'Sector': row['GICS Sector'],
                                'Ticker': ticker,
                                'Name': row['Security'],
                                'Undervaluation %': undervaluation,
                                'Market Cap (B)': data.get('market_cap') / 1e9 if data.get('market_cap') else None,
                                'Intrinsic Value': valuation_results.get('intrinsic_value'),
                                'Beta': data.get('beta')
                            })
                except Exception as e:
                    st.warning(f"Error processing {ticker}: {str(e)}")
                progress_bar.progress(min((i + idx + 1) / total_stocks, 1.0))

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Undervaluation %', ascending=False, na_position='last')

    return results_df

def display_screener():
    """
    Display the stock screener interface in Streamlit.
    """
    st.header("S&P 500 Stock Screener")
    st.markdown("Screen S&P 500 stocks for undervaluation based on the selected valuation model.")

    try:
        sp500_data = get_sp500_tickers()
        sector_options = sorted(sp500_data['GICS Sector'].unique())
    except Exception as e:
        st.error(f"Error loading sector options: {str(e)}")
        return

    model = st.selectbox(
        "Valuation Model",
        ["Core Valuation (Excel)", "Lynch Method", "Discounted Cash Flow (DCF)",
         "Dividend Discount Model (DDM)", "Two-Stage DCF", "Residual Income (RI)",
         "Reverse DCF", "Graham Intrinsic Value"],
        key="screener_model"
    )

    min_undervaluation = st.number_input(
        "Minimum Undervaluation %",
        min_value=-100.0, max_value=100.0, value=0.0, step=0.1,
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
        ["Undervaluation % (Descending)", "Market Cap (Descending)", "Beta (Ascending)"],
        help="Sort results by undervaluation, market cap, or beta."
    )

    max_stocks = st.number_input(
        "Max Stocks to Screen",
        min_value=1, max_value=500, value=500, step=10,
        help="Limit the number of stocks to process to manage performance."
    )

    if st.button("Run Screener"):
        with st.spinner("Screening S&P 500 stocks..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results_df = loop.run_until_complete(
                    run_screener_async(model, min_undervaluation, selected_sectors, max_stocks)
                )
                loop.close()

                if results_df.empty:
                    st.info("No undervalued stocks found with the given criteria.")
                else:
                    if sort_by == "Market Cap (Descending)":
                        results_df = results_df.sort_values(by='Market Cap (B)', ascending=False, na_position='last')
                    elif sort_by == "Beta (Ascending)":
                        results_df = results_df.sort_values(by='Beta', ascending=True, na_position='last')
                    st.dataframe(results_df, use_container_width=True)

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Screener Results",
                        data=csv,
                        file_name="sp500_screener_results.csv",
                        mime="text/csv"
                    )

                    # Portfolio integration
                    if not st.session_state.get('portfolio', pd.DataFrame()).empty:
                        st.subheader("Add to Portfolio")
                        selected_tickers = st.multiselect("Select Tickers to Add to Portfolio", results_df['Ticker'].tolist())
                        if st.button("Add Selected to Portfolio"):
                            for ticker in selected_tickers:
                                row = results_df[results_df['Ticker'] == ticker].iloc[0]
                                new_row = pd.DataFrame([{
                                    'Ticker': ticker,
                                    'Intrinsic Value': row['Intrinsic Value'],
                                    'Undervaluation %': row['Undervaluation %'],
                                    'Verdict': 'Buy' if row['Undervaluation %'] > 0 else 'Hold' if row['Undervaluation %'] > -20 else 'Sell',
                                    'Beta': row['Beta']
                                }]).astype(st.session_state.portfolio.dtypes)
                                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                            st.success(f"Added {len(selected_tickers)} stocks to portfolio.")

            except Exception as e:
                st.error(f"Error running screener: {str(e)}")

if __name__ == "__main__":
    display_screener()
