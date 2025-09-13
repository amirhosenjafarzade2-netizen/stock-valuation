import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import yfinance as yf
from valuation_models import calculate_valuation
from data_fetch import fetch_stock_data
from visualizations import plot_heatmap, plot_monte_carlo, plot_model_comparison
from utils import validate_inputs, export_portfolio, generate_pdf_report
from monte_carlo import run_monte_carlo
from graphs import display_fundamental_graphs
from screener import display_screener

# Set Streamlit page config
st.set_page_config(page_title="Stock Valuation Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta'])
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

# Load custom CSS
with open("styles.html") as f:
    st.html(f.read())

# Custom CSS for red tab headers (always red)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        flex-direction: row !important;
        padding: 0px !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        background-color: #f0f2f6 !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        border: none !important;
        color: #d32f2f !important;
        font-weight: 600 !important;
        flex: 1 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 0 12px !important;
        white-space: nowrap !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #b71c1c !important;
        background-color: #ffebee !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #d32f2f !important;
        background-color: #fff !important;
        box-shadow: inset 0 -2px 0 #d32f2f !important;
    }
</style>
""", unsafe_allow_html=True)

# Main UI with title and tabs
st.title("Stock Valuation Dashboard")
st.markdown("Analyze stocks using valuation models, view fundamental graphs, or screen the S&P 500. *Not financial advice. Verify all inputs and calculations independently.*")

# Create tabs: Valuation Dashboard, Fundamental Graphs, S&P 500 Screener
tab1, tab2, tab3 = st.tabs(["Valuation Dashboard", "Fundamental Graphs", "S&P 500 Screener"])

with tab1:
    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Parameters")
        
        with st.expander("Model Descriptions"):
            st.markdown("""
            - **Core Valuation (Excel)**: Best for steady-growth sectors (consumer goods, industrials, healthcare).
            - **Lynch Method**: Ideal for high-growth sectors (tech, biotech, consumer discretionary).
            - **DCF**: Versatile, excels in capital-intensive sectors (tech, pharma, energy).
            - **DDM**: Best for dividend-paying sectors (utilities, consumer staples, REITs, financials).
            - **Two-Stage DCF**: Suited for transitional firms (tech, biotech) with high-to-stable growth.
            - **Residual Income (RI)**: Best for asset-heavy sectors (financials, real estate, utilities).
            - **Reverse DCF**: Useful across sectors, especially growth stocks (tech, healthcare).
            - **Graham Intrinsic Value**: Best for value investing in undervalued stocks (any sector).
            """)
        
        model = st.selectbox(
            "Valuation Model",
            ["Core Valuation (Excel)", "Lynch Method", "Discounted Cash Flow (DCF)", "Dividend Discount Model (DDM)",
             "Two-Stage DCF", "Residual Income (RI)", "Reverse DCF", "Graham Intrinsic Value"],
            help="Select a model to analyze the stock."
        )
        
        ticker = st.text_input("Ticker Symbol", help="Enter a valid ticker (e.g., AAPL) to fetch data or input manually.")
        if st.button("Fetch"):
            try:
                data = fetch_stock_data(ticker)
                st.session_state.data = data
                st.success(f"Data fetched for {ticker}")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}. Enter manually.")
        
        data = st.session_state.get('data', {})
        
        def safe_value(key, default, min_val=None, max_val=None):
            val = data.get(key, default)
            if min_val is not None and val < min_val:
                val = min_val
            if max_val is not None and val > max_val:
                val = max_val
            return val
        
        desired_return = st.number_input("Desired Return/Cost of Equity (%)", min_value=0.0, max_value=50.0, value=safe_value('desired_return', 10.0), key="desired_return", help="Must be 0-50%")
        years_high_growth = st.number_input("Years (High-Growth)", min_value=3, max_value=10, value=safe_value('years_high_growth', 5), step=1, key="years_high_growth", help="Must be 3-10 years")
        current_price = st.number_input("Current Price", min_value=0.01, value=safe_value('current_price', 100.0, min_val=0.01), key="current_price", help="Must be positive")
        current_eps = st.number_input("Current EPS (TTM)", value=safe_value('current_eps', 5.0), key="current_eps", help="Must be non-zero for Core/Lynch/RI/Graham. Leave blank for DCF with FCF.")
        forward_eps = st.number_input("Forward EPS (Next Yr)", min_value=0.01, value=safe_value('forward_eps', 5.5, min_val=0.01), key="forward_eps", help="Must be positive for Core/Lynch")
        historical_pe = st.number_input("Historical Avg P/E", min_value=0.01, value=safe_value('historical_pe', 15.0, min_val=0.01), key="historical_pe", help="Must be positive")
        analyst_growth = st.number_input("Analyst Growth (5y, %)", min_value=0.0, max_value=50.0, value=safe_value('analyst_growth', 10.0), key="analyst_growth", help="Must be 0-50%")
        exit_pe = st.number_input("Exit P/E", min_value=0.01, value=safe_value('exit_pe', 15.0, min_val=0.01), key="exit_pe", help="Must be positive. Defaults to Historical Avg P/E")
        core_mos = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=safe_value('core_mos', 25.0), key="core_mos", help="Must be 0-100%")
        
        with st.expander("Dividend Inputs"):
            dividend_per_share = st.number_input("Current Dividend Per Share", min_value=0.0, value=safe_value('dividend_per_share', 1.0), key="dividend_per_share")
            dividend_growth = st.number_input("Dividend Growth (5y, %)", min_value=0.0, max_value=50.0, value=safe_value('dividend_growth', 5.0), key="dividend_growth")
            dividend_mos = st.number_input("Dividend MOS (%)", min_value=0.0, max_value=100.0, value=safe_value('dividend_mos', 25.0), key="dividend_mos")
        
        with st.expander("DCF Inputs"):
            fcf = st.number_input("Free Cash Flow (Latest, $M)", min_value=0.0, value=safe_value('fcf', 0.0), key="fcf")
            stable_growth = st.number_input("Stable Growth Rate (%)", min_value=0.0, max_value=50.0, value=safe_value('stable_growth', 3.0), key="stable_growth")
            tax_rate = st.number_input("Tax Rate (%)", min_value=0.0, max_value=100.0, value=safe_value('tax_rate', 25.0), key="tax_rate")
            wacc = st.number_input("WACC (%)", min_value=0.0, max_value=50.0, value=safe_value('wacc', 8.0), key="wacc")
            dcf_mos = st.number_input("DCF MOS (%)", min_value=0.0, max_value=100.0, value=safe_value('dcf_mos', 25.0), key="dcf_mos")
        
        with st.expander("RI/Graham Inputs"):
            book_value = st.number_input("Book Value Per Share", min_value=0.01, value=safe_value('book_value', 20.0, min_val=0.01), key="book_value")
            roe = st.number_input("Return on Equity (%)", min_value=0.0, max_value=100.0, value=safe_value('roe', 15.0), key="roe")
            ri_mos = st.number_input("RI MOS (%)", min_value=0.0, max_value=100.0, value=safe_value('ri_mos', 25.0), key="ri_mos")
        
        with st.expander("Portfolio & Monte Carlo"):
            beta = st.number_input("Stock Beta", min_value=0.0, max_value=10.0, value=safe_value('beta', 1.0), key="beta")
            add_to_portfolio = st.checkbox("Add to Portfolio", value=False, key="add_to_portfolio")
            export = st.checkbox("Export Portfolio", value=False, key="export")
            monte_carlo_runs = st.number_input("Monte Carlo Runs", min_value=100, max_value=10000, value=safe_value('monte_carlo_runs', 1000), step=100, key="monte_carlo_runs")
            growth_adj = st.number_input("Growth Adjustment (%)", min_value=0.0, max_value=50.0, value=safe_value('growth_adj', 10.0), key="growth_adj")
            wacc_adj = st.number_input("WACC Adjustment (%)", min_value=0.0, max_value=50.0, value=safe_value('wacc_adj', 10.0), key="wacc_adj")
    
    # Main content
    col_left, col_right = st.columns([2, 3])
    
    with col_left:
        st.header("Valuation Dashboard")
        # Reactive calculations
        inputs = {
            'model': model, 'ticker': ticker, 'desired_return': desired_return, 'years_high_growth': years_high_growth,
            'current_price': current_price, 'current_eps': current_eps, 'forward_eps': forward_eps,
            'historical_pe': historical_pe, 'analyst_growth': analyst_growth, 'exit_pe': exit_pe,
            'core_mos': core_mos, 'dividend_per_share': dividend_per_share, 'dividend_growth': dividend_growth,
            'dividend_mos': dividend_mos, 'fcf': fcf, 'stable_growth': stable_growth, 'tax_rate': tax_rate,
            'wacc': wacc, 'dcf_mos': dcf_mos, 'book_value': book_value, 'roe': roe, 'ri_mos': ri_mos,
            'beta': beta
        }
        if validate_inputs(inputs):
            results = calculate_valuation(inputs)
            st.session_state.results = results
        else:
            st.error("Invalid inputs. Check requirements.")
            results = {}
        
        # Display results
        st.metric("Model", model)
        st.metric("Current Price", f"${results.get('current_price', current_price):.2f}")
        st.metric("Intrinsic Value (Today)", f"${results.get('intrinsic_value', 0):.2f}")
        st.metric("Safe Buy Price (after MOS)", f"${results.get('safe_buy_price', 0):.2f}")
        st.metric("Undervaluation %", f"${results.get('undervaluation', 0):.2f}%")
        st.metric("Implied Growth (Reverse DCF)", f"{results.get('implied_growth', 0):.2f}%")
        st.metric("Required EPS CAGR", f"{results.get('eps_cagr', 0):.2f}%")
        st.metric("PEG Ratio", f"{results.get('peg_ratio', 0):.2f}")
        st.metric("Forward vs Historical P/E Delta", f"{results.get('pe_delta', 0):.2f}%")
        st.metric("Overall Score (0-100)", f"{results.get('score', 0)}")
        st.metric("Verdict", results.get('verdict', '-'))
        st.metric("Lynch Fair Value", f"${results.get('lynch_value', 0):.2f}")
        st.metric("DCF Intrinsic Value", f"${results.get('dcf_value', 0):.2f}")
        st.metric("DDM Intrinsic Value", f"${results.get('ddm_value', 0):.2f}")
        st.metric("Two-Stage DCF Value", f"${results.get('two_stage_dcf', 0):.2f}")
        st.metric("Residual Income Value", f"${results.get('ri_value', 0):.2f}")
        st.metric("Graham Intrinsic Value", f"${results.get('graham_value', 0):.2f}")
    
    with col_right:
        st.header("Portfolio Overview")
        portfolio_beta = st.session_state.portfolio['Beta'].mean() if not st.session_state.portfolio.empty else 0
        expected_return = portfolio_beta * 8.0
        st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        st.metric("Portfolio Expected Return", f"{expected_return:.2f}%")
        
        if add_to_portfolio and 'results' in st.session_state:
            new_row = pd.DataFrame([{
                'Ticker': ticker,
                'Intrinsic Value': results.get('intrinsic_value', 0),
                'Undervaluation %': results.get('undervaluation', 0),
                'Verdict': results.get('verdict', '-'),
                'Beta': beta
            }])
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
        
        st.dataframe(st.session_state.portfolio, use_container_width=True)
        
        if export:
            export_portfolio(st.session_state.portfolio, "portfolio.csv")
            st.success("Portfolio exported as portfolio.csv")
        
        st.header("Scenario Analysis")
        with st.expander("Adjust Scenarios"):
            bull_adj = st.slider("Bull Case Adjustment (%)", -50.0, 50.0, 20.0, key="bull_adj")
            bear_adj = st.slider("Bear Case Adjustment (%)", -50.0, 50.0, -20.0, key="bear_adj")
        scenarios = pd.DataFrame({
            'Scenario': ['Base Case', 'Bull Case', 'Bear Case'],
            'Intrinsic Value': [results.get('intrinsic_value', 0), results.get('intrinsic_value', 0) * (1 + bull_adj/100), results.get('intrinsic_value', 0) * (1 + bear_adj/100)],
            'Undervaluation %': [results.get('undervaluation', 0), results.get('undervaluation', 0) + bull_adj, results.get('undervaluation', 0) + bear_adj]
        })
        st.dataframe(scenarios, use_container_width=True)

        st.header("Sensitivity Analysis (Heatmap)")
        heatmap = plot_heatmap(results.get('intrinsic_value', 0), wacc, analyst_growth)
        st.plotly_chart(heatmap, use_container_width=True)

        st.header("Monte Carlo Simulation")
        mc_results = run_monte_carlo(inputs, monte_carlo_runs, growth_adj, wacc_adj)
        st.metric("Average Intrinsic Value", f"${mc_results.get('avg_value', 0):.2f}")
        st.metric("Std Dev", f"${mc_results.get('std_dev', 0):.2f}")
        st.metric("Probability Undervalued (> Current Price)", f"{mc_results.get('prob_undervalued', 0):.2f}%")
        mc_plot = plot_monte_carlo(mc_results)
        st.plotly_chart(mc_plot, use_container_width=True)

        st.header("Model Comparison")
        model_comp = pd.DataFrame({
            'Model': ['Core', 'Lynch', 'DCF', 'DDM', 'Two-Stage DCF', 'RI', 'Reverse DCF', 'Graham'],
            'Intrinsic Value': [
                results.get('core_value', 0), results.get('lynch_value', 0), results.get('dcf_value', 0),
                results.get('ddm_value', 0), results.get('two_stage_dcf', 0), results.get('ri_value', 0),
                results.get('reverse_dcf_value', 0), results.get('graham_value', 0)
            ]
        })
        comp_plot = plot_model_comparison(model_comp)
        st.plotly_chart(comp_plot, use_container_width=True)

with tab2:
    st.header("Fundamental Graphs")
    ticker_graphs = st.text_input("Enter Ticker for Graphs", value=st.session_state.get('data', {}).get('ticker', ''), help="Enter a ticker (e.g., AAPL) to view fundamental graphs.")
    if st.button("Fetch Graphs Data"):
        if ticker_graphs:
            try:
                display_fundamental_graphs(ticker_graphs)
                st.success(f"Graphs loaded for {ticker_graphs}")
            except Exception as e:
                st.error(f"Error loading graphs: {str(e)}")
    elif ticker_graphs:
        display_fundamental_graphs(ticker_graphs)

with tab3:
    display_screener()

st.markdown("---")
st.markdown("*Disclaimer: This tool is for informational purposes only and not financial advice. Verify all inputs and calculations independently.*")
