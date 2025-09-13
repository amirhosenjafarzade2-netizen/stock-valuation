import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import yfinance as yf  # For data fetching (stubbed for now)
# Placeholder imports for other modules
from valuation_models import calculate_valuation
from data_fetch import fetch_stock_data
from visualizations import plot_heatmap, plot_monte_carlo, plot_model_comparison
from utils import validate_inputs, export_portfolio, generate_pdf_report
from monte_carlo import run_monte_carlo

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

# Main UI
st.title("Stock Valuation Dashboard")
st.markdown("Analyze stocks using valuation models: Core, Lynch, DCF, DDM, Two-Stage DCF, Residual Income, Reverse DCF, Graham Intrinsic Value.")
st.markdown("*For informational purposes only. Not financial advice. Verify all inputs and calculations independently.*")

# Theme toggle
theme = st.checkbox("Dark Mode", value=False, key="theme")
if theme:
    st.markdown('<style>body {background-color: #1E1E1E; color: #FFFFFF;}</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body {background-color: #FFFFFF; color: #000000;}</style>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    
    # Model Descriptions
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
    
    # Valuation Model Selection
    model = st.selectbox(
        "Valuation Model",
        ["Core Valuation (Excel)", "Lynch Method", "Discounted Cash Flow (DCF)", "Dividend Discount Model (DDM)",
         "Two-Stage DCF", "Residual Income (RI)", "Reverse DCF", "Graham Intrinsic Value"],
        help="Select a model to analyze the stock."
    )
    
    # Ticker and Data Fetching
    ticker = st.text_input("Ticker Symbol", help="Enter a valid ticker (e.g., AAPL) to fetch data or input manually.")
    if st.button("Fetch"):
        try:
            data = fetch_stock_data(ticker)
            st.session_state.data = data
            st.success(f"Data fetched for {ticker}")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}. Enter manually.")
    
    # Initialize inputs with fetched data or defaults
    data = st.session_state.get('data', {})
    desired_return = st.number_input("Desired Return/Cost of Equity (%)", min_value=0.0, max_value=50.0, value=data.get('desired_return', 10.0), key="desired_return", help="Must be 0-50%")
    years_high_growth = st.number_input("Years (High-Growth)", min_value=3, max_value=10, value=data.get('years_high_growth', 5), step=1, key="years_high_growth", help="Must be 3-10 years")
    current_price = st.number_input("Current Price", min_value=0.01, value=data.get('current_price', 100.0), key="current_price", help="Must be positive")
    current_eps = st.number_input("Current EPS (TTM)", value=data.get('current_eps', 5.0), key="current_eps", help="Must be non-zero for Core/Lynch/RI/Graham. Leave blank for DCF with FCF.")
    forward_eps = st.number_input("Forward EPS (Next Yr)", min_value=0.01, value=data.get('forward_eps', 5.5), key="forward_eps", help="Must be positive for Core/Lynch")
    historical_pe = st.number_input("Historical Avg P/E", min_value=0.01, value=data.get('historical_pe', 15.0), key="historical_pe", help="Must be positive")
    analyst_growth = st.number_input("Analyst Growth (5y, %)", min_value=0.0, max_value=50.0, value=data.get('analyst_growth', 10.0), key="analyst_growth", help="Must be 0-50%")
    exit_pe = st.number_input("Exit P/E", min_value=0.01, value=data.get('exit_pe', historical_pe), key="exit_pe", help="Must be positive. Defaults to Historical Avg P/E")
    core_mos = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=data.get('core_mos', 25.0), key="core_mos", help="Must be 0-100%")
    
    # Dividend Inputs
    with st.expander("Dividend Inputs"):
        dividend_per_share = st.number_input("Current Dividend Per Share", min_value=0.0, value=data.get('dividend_per_share', 1.0), key="dividend_per_share", help="Must be positive for DDM")
        dividend_growth = st.number_input("Dividend Growth Rate (%)", min_value=0.0, max_value=50.0, value=data.get('dividend_growth', 5.0), key="dividend_growth", help="Must be 0-50%. Default 5% if dividends present.")
        dividend_mos = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=data.get('dividend_mos', 25.0), key="dividend_mos", help="Must be 0-100%")
    
    # DCF & Reverse DCF Inputs
    with st.expander("DCF & Reverse DCF Inputs"):
        fcf = st.number_input("Free Cash Flow (FCF, optional)", value=data.get('fcf', 0.0), key="fcf", help="Must be non-zero for DCF if no EPS")
        stable_growth = st.number_input("Stable Growth Rate (%)", min_value=0.0, max_value=50.0, value=data.get('stable_growth', 3.0), key="stable_growth", help="Must be 0-50% and < WACC")
        tax_rate = st.number_input("Tax Rate (%)", min_value=0.0, max_value=100.0, value=data.get('tax_rate', 25.0), key="tax_rate", help="Must be 0-100%")
        wacc = st.number_input("WACC (%)", min_value=0.0, max_value=50.0, value=data.get('wacc', 8.0), key="wacc", help="Must be 0-50% and > Stable Growth")
        dcf_mos = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=data.get('dcf_mos', 25.0), key="dcf_mos", help="Must be 0-100%")
    
    # Residual Income Inputs
    with st.expander("Residual Income Inputs"):
        book_value = st.number_input("Book Value Per Share", min_value=0.01, value=data.get('book_value', 20.0), key="book_value", help="Must be positive for RI/Graham")
        roe = st.number_input("ROE (%)", min_value=0.0, max_value=100.0, value=data.get('roe', 15.0), key="roe", help="Must be 0-100%")
        ri_mos = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=data.get('ri_mos', 25.0), key="ri_mos", help="Must be 0-100%")
    
    # Portfolio Beta Input
    beta = st.number_input("Beta (for Portfolio)", min_value=0.0, value=data.get('beta', 1.0), key="beta", help="Used for portfolio calculations")
    
    # Monte Carlo Settings
    with st.expander("Monte Carlo Settings"):
        monte_carlo_runs = st.number_input("Number of Runs", min_value=100, max_value=2000, value=data.get('monte_carlo_runs', 1000), step=100, key="monte_carlo_runs", help="100-2000 runs (lower for faster performance)")
        growth_adj = st.number_input("Growth Adjustment Range (±%)", min_value=0.0, max_value=50.0, value=data.get('growth_adj', 10.0), key="growth_adj", help="0-50%")
        wacc_adj = st.number_input("WACC Adjustment Range (±%)", min_value=0.0, max_value=50.0, value=data.get('wacc_adj', 10.0), key="wacc_adj", help="Must be 0-50%")
    
    # Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        add_to_portfolio = st.button("Add to Portfolio")
    with col2:
        export = st.button("Export Portfolio")
    download_report = st.button("Download Report")

# Main Dashboard Layout
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
    if validate_inputs(inputs):  # From utils.py
        results = calculate_valuation(inputs)  # From valuation_models.py
        st.session_state.results = results
    else:
        st.error("Invalid inputs. Check requirements.")
        results = {}
    
    # Display results
    st.metric("Model", model)
    st.metric("Current Price", f"${results.get('current_price', current_price):.2f}")
    st.metric("Intrinsic Value (Today)", f"${results.get('intrinsic_value', 0):.2f}")
    st.metric("Safe Buy Price (after MOS)", f"${results.get('safe_buy_price', 0):.2f}")
    st.metric("Undervaluation %", f"{results.get('undervaluation', 0):.2f}%")
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
    expected_return = portfolio_beta * 8.0  # Simplified CAPM
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
        bull_adj = st.slider("Bull Case Adjustment (%)", -50.0, 50.0, 20.0)
        bear_adj = st.slider("Bear Case Adjustment (%)", -50.0, 50.0, -20.0)
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

st.markdown("---")
st.markdown("*Disclaimer: This tool is for informational purposes only and not financial advice. Verify all inputs and calculations independently.*")
