import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import yfinance as yf  # For data fetching (stubbed for now)
# Placeholder imports for other modules (to be implemented)
from valuation_models import calculate_valuation
from data_fetch import fetch_stock_data
from visualizations import plot_heatmap, plot_monte_carlo, plot_model_comparison
from utils import validate_inputs, export_portfolio, generate_pdf_report
from monte_carlo import run_monte_carlo

# Set Streamlit page config
st.set_page_config(page_title="Stock Valuation Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta'])

# Load custom CSS
with open("styles.html") as f:
    st.html(f.read())

# Main UI
st.title("Stock Valuation Dashboard")
st.markdown("Analyze stocks using valuation models: Core, Lynch, DCF, DDM, Two-Stage DCF, Residual Income, Reverse DCF, Graham Intrinsic Value.")
st.markdown("*For informational purposes only. Not financial advice. Verify all inputs and calculations independently.*")

# Theme toggle
theme = st.checkbox("Dark Mode", value=False)
if theme:
    st.markdown('<style>body {background-color: #1E1E1E; color: #FFFFFF;}</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body {background-color: #FFFFFF; color: #000000;}</style>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    
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
            data = fetch_stock_data(ticker)  # Placeholder call
            st.session_state.data = data
            st.success(f"Data fetched for {ticker}")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}. Enter manually.")
    
    # Core Inputs
    with st.expander("Core Inputs"):
        desired_return = st.number_input("Desired Return/Cost of Equity (%)", min_value=0.0, max_value=50.0, value=10.0, help="Must be 0-50%")
        years_high_growth = st.number_input("Years (High-Growth)", min_value=3, max_value=10, value=5, step=1, help="Must be 3-10 years")
        current_price = st.number_input("Current Price", min_value=0.01, value=100.0, help="Must be positive")
        current_eps = st.number_input("Current EPS (TTM)", value=5.0, help="Must be non-zero for Core/Lynch/RI/Graham. Leave blank for DCF with FCF.")
        forward_eps = st.number_input("Forward EPS (Next Yr)", min_value=0.01, value=5.5, help="Must be positive for Core/Lynch")
        historical_pe = st.number_input("Historical Avg P/E", min_value=0.01, value=15.0, help="Must be positive")
        analyst_growth = st.number_input("Analyst Growth (5y, %)", min_value=0.0, max_value=50.0, value=10.0, help="Must be 0-50%")
        exit_pe = st.number_input("Exit P/E", min_value=0.01, value=historical_pe, help="Must be positive. Defaults to Historical Avg P/E")
        core_mos = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=25.0, help="Must be 0-100%")
    
    # Dividend Inputs
    with st.expander("Dividend Inputs"):
        dividend_per_share = st.number_input("Current Dividend Per Share", min_value=0.0, value=1.0, help="Must be positive for DDM")
        dividend_growth = st.number_input("Dividend Growth Rate (%)", min_value=0.0, max_value=50.0, value=5.0, help="Must be 0-50%. Default 5% if dividends present.")
        dividend_mos = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=25.0, help="Must be 0-100%")
    
    # DCF & Reverse DCF Inputs
    with st.expander("DCF & Reverse DCF Inputs"):
        fcf = st.number_input("Free Cash Flow (FCF, optional)", value=0.0, help="Must be non-zero for DCF if no EPS")
        stable_growth = st.number_input("Stable Growth Rate (%)", min_value=0.0, max_value=50.0, value=3.0, help="Must be 0-50% and < WACC")
        tax_rate = st.number_input("Tax Rate (%)", min_value=0.0, max_value=100.0, value=25.0, help="Must be 0-100%")
        wacc = st.number_input("WACC (%)", min_value=0.0, max_value=50.0, value=8.0, help="Must be 0-50% and > Stable Growth")
        dcf_mos = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=25.0, help="Must be 0-100%")
    
    # Residual Income Inputs
    with st.expander("Residual Income Inputs"):
        book_value = st.number_input("Book Value Per Share", min_value=0.01, value=20.0, help="Must be positive for RI/Graham")
        roe = st.number_input("ROE (%)", min_value=0.0, max_value=100.0, value=15.0, help="Must be 0-100%")
        ri_mos = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=25.0, help="Must be 0-100%")
    
    # Monte Carlo Settings
    with st.expander("Monte Carlo Settings"):
        monte_carlo_runs = st.number_input("Number of Runs", min_value=100, max_value=2000, value=1000, step=100, help="100-2000 runs (lower for faster performance)")
        growth_adj = st.number_input("Growth Adjustment Range (±%)", min_value=0.0, max_value=50.0, value=10.0, help="0-50%")
        wacc_adj = st.number_input("WACC Adjustment Range (±%)", min_value=0.0, max_value=50.0, value=10.0, help="0-50%")
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        calculate = st.button("Calculate")
    with col2:
        add_to_portfolio = st.button("Add to Portfolio")
    with col3:
        export = st.button("Export Portfolio")
    download_report = st.button("Download Report")

# Main Dashboard Layout
col_left, col_right = st.columns([2, 3])

with col_left:
    st.header("Valuation Dashboard")
    if calculate:
        # Validate inputs
        inputs = {
            'model': model, 'ticker': ticker, 'desired_return': desired_return, 'years_high_growth': years_high_growth,
            'current_price': current_price, 'current_eps': current_eps, 'forward_eps': forward_eps,
            'historical_pe': historical_pe, 'analyst_growth': analyst_growth, 'exit_pe': exit_pe,
            'core_mos': core_mos, 'dividend_per_share': dividend_per_share, 'dividend_growth': dividend_growth,
            'dividend_mos': dividend_mos, 'fcf': fcf, 'stable_growth': stable_growth, 'tax_rate': tax_rate,
            'wacc': wacc, 'dcf_mos': dcf_mos, 'book_value': book_value, 'roe': roe, 'ri_mos': ri_mos
        }
        if validate_inputs(inputs):  # Placeholder validation
            results = calculate_valuation(inputs)  # Placeholder call
            st.session_state.results = results
        else:
            st.error("Invalid inputs. Check requirements.")
    
    # Display results (stubbed)
    results = st.session_state.get('results', {})
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
    beta = st.session_state.portfolio['Beta'].mean() if not st.session_state.portfolio.empty else 0
    expected_return = beta * 8.0  # Simplified CAPM (risk-free rate + beta * market return)
    st.metric("Portfolio Beta", f"{beta:.2f}")
    st.metric("Portfolio Expected Return", f"{expected_return:.2f}%")
    
    if add_to_portfolio and 'results' in st.session_state:
        new_row = pd.DataFrame([{
            'Ticker': ticker,
            'Intrinsic Value': results.get('intrinsic_value', 0),
            'Undervaluation %': results.get('undervaluation', 0),
            'Verdict': results.get('verdict', '-'),
            'Beta': st.session_state.get('data', {}).get('beta', 1.0)
        }])
        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
    
    st.dataframe(st.session_state.portfolio, use_container_width=True)
    
    if export:
        export_portfolio(st.session_state.portfolio, "portfolio.csv")
        st.success("Portfolio exported as portfolio.csv")
    
    if download_report:
        pdf_file = generate_pdf_report(st.session_state.get('results', {}), st.session_state.portfolio)
        st.download_button("Download PDF Report", data=pdf_file, file_name=f"valuation_report_{ticker}_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")

    st.header("Scenario Analysis")
    scenarios = pd.DataFrame({
        'Scenario': ['Base Case', 'Bull Case', 'Bear Case'],
        'Intrinsic Value': [results.get('intrinsic_value', 0), results.get('intrinsic_value', 0) * 1.2, results.get('intrinsic_value', 0) * 0.8],
        'Undervaluation %': [results.get('undervaluation', 0), results.get('undervaluation', 0) + 10, results.get('undervaluation', 0) - 10]
    })
    st.dataframe(scenarios, use_container_width=True)

    st.header("Sensitivity Analysis (Heatmap)")
    heatmap = plot_heatmap(results.get('intrinsic_value', 0), wacc, analyst_growth)  # Placeholder
    st.plotly_chart(heatmap, use_container_width=True)

    st.header("Monte Carlo Simulation")
    if calculate:
        mc_results = run_monte_carlo(inputs, monte_carlo_runs, growth_adj, wacc_adj)  # Placeholder
        st.metric("Average Intrinsic Value", f"${mc_results.get('avg_value', 0):.2f}")
        st.metric("Std Dev", f"${mc_results.get('std_dev', 0):.2f}")
        st.metric("Probability Undervalued (> Current Price)", f"{mc_results.get('prob_undervalued', 0):.2f}%")
        mc_plot = plot_monte_carlo(mc_results)  # Placeholder
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
    comp_plot = plot_model_comparison(model_comp)  # Placeholder
    st.plotly_chart(comp_plot, use_container_width=True)

st.markdown("---")
st.markdown("*Disclaimer: This tool is for informational purposes only and not financial advice. Verify all inputs and calculations independently.*")
