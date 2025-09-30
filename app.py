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

st.set_page_config(page_title="Stock Valuation Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize session state with type coercion
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta']).astype({
        'Ticker': str, 'Intrinsic Value': float, 'Undervaluation %': float, 'Verdict': str, 'Beta': float
    })
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

try:
    with open("styles.html") as f:
        st.html(f.read())
except FileNotFoundError:
    st.warning("styles.html not found. Using default styling.")

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

st.title("Stock Valuation Dashboard")
st.markdown("Analyze stocks using valuation models, view fundamental graphs, or screen the S&P 500. *Not financial advice.*")

tab1, tab2, tab3 = st.tabs(["Valuation Dashboard", "Fundamental Graphs", "S&P 500 Screener"])

with tab1:
    with st.sidebar:
        st.header("Input Parameters")
        
        with st.expander("Model Descriptions"):
            st.markdown("""
            - **Core Valuation (Excel)**: Steady-growth sectors.
            - **Lynch Method**: High-growth sectors.
            - **DCF**: Capital-intensive sectors.
            - **DDM**: Dividend-paying sectors.
            - **Two-Stage DCF**: Transitional firms.
            - **Residual Income (RI)**: Asset-heavy sectors.
            - **Reverse DCF**: Growth stocks.
            - **Graham Intrinsic Value**: Value investing.
            """)
        
        model = st.selectbox(
            "Valuation Model",
            ["Core Valuation (Excel)", "Lynch Method", "Discounted Cash Flow (DCF)", "Dividend Discount Model (DDM)",
             "Two-Stage DCF", "Residual Income (RI)", "Reverse DCF", "Graham Intrinsic Value"],
            help="Select a model to analyze the stock."
        )
        
        ticker = st.text_input("Ticker Symbol", help="Enter a valid ticker (e.g., AAPL).")
        if st.button("Fetch"):
            try:
                data = fetch_stock_data(ticker)
                st.session_state.data = data
                st.success(f"Data fetched for {ticker}")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}. Enter manually.")
    
    # Truncated portion assumed to include valuation logic
    # Assuming inputs are constructed and results calculated
    inputs = st.session_state.data.copy()
    inputs['model'] = model
    results = calculate_valuation(inputs)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.header("Valuation Results")
        if 'error' in results:
            st.error(results['error'])
        else:
            st.metric("Current Price", f"${results.get('current_price', 0):.2f}")
            st.metric("Intrinsic Value", f"${results.get('intrinsic_value', 0):.2f}" if results.get('intrinsic_value') else "N/A")
            st.metric("Safe Buy Price", f"${results.get('safe_buy_price', 0):.2f}" if results.get('safe_buy_price') else "N/A")
            st.metric("Undervaluation %", f"{results.get('undervaluation', 0):.2f}%" if results.get('undervaluation') else "N/A")
            st.metric("Overall Score (0-100)", f"{results.get('score', 0)}" if results.get('score') else "N/A")
            st.metric("Verdict", results.get('verdict', 'N/A'))
            st.metric("Lynch Fair Value", f"${results.get('lynch_value', 0):.2f}" if results.get('lynch_value') else "N/A")
            st.metric("DCF Intrinsic Value", f"${results.get('dcf_value', 0):.2f}" if results.get('dcf_value') else "N/A")
            st.metric("DDM Intrinsic Value", f"${results.get('ddm_value', 0):.2f}" if results.get('ddm_value') else "N/A")
            st.metric("Two-Stage DCF Value", f"${results.get('two_stage_dcf', 0):.2f}" if results.get('two_stage_dcf') else "N/A")
            st.metric("Residual Income Value", f"${results.get('ri_value', 0):.2f}" if results.get('ri_value') else "N/A")
            st.metric("Graham Intrinsic Value", f"${results.get('graham_value', 0):.2f}" if results.get('graham_value') else "N/A")
    
    with col_right:
        st.header("Portfolio Overview")
        portfolio_beta = st.session_state.portfolio['Beta'].mean() if not st.session_state.portfolio.empty else 0
        expected_return = portfolio_beta * 8.0
        st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        st.metric("Portfolio Expected Return", f"{expected_return:.2f}%")
        
        add_to_portfolio = st.button("Add to Portfolio")
        if add_to_portfolio and 'results' in st.session_state:
            new_row = pd.DataFrame([{
                'Ticker': ticker,
                'Intrinsic Value': results.get('intrinsic_value', None),
                'Undervaluation %': results.get('undervaluation', None),
                'Verdict': results.get('verdict', 'N/A'),
                'Beta': inputs.get('beta', 1.0)
            }]).astype(st.session_state.portfolio.dtypes)  # CHANGED: Type coercion
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
        
        st.dataframe(st.session_state.portfolio, use_container_width=True)
        
        if st.button("Export Portfolio"):
            export_portfolio(st.session_state.portfolio, "portfolio.csv")
            st.success("Portfolio exported as portfolio.csv")
        
        st.header("Scenario Analysis")
        with st.expander("Adjust Scenarios"):
            bull_adj = st.slider("Bull Case Adjustment (%)", -50.0, 50.0, 20.0, key="bull_adj")
            bear_adj = st.slider("Bear Case Adjustment (%)", -50.0, 50.0, -20.0, key="bear_adj")
        scenarios = pd.DataFrame({
            'Scenario': ['Base Case', 'Bull Case', 'Bear Case'],
            'Intrinsic Value': [
                results.get('intrinsic_value', None),
                results.get('intrinsic_value', None) * (1 + bull_adj/100) if results.get('intrinsic_value') else None,
                results.get('intrinsic_value', None) * (1 + bear_adj/100) if results.get('intrinsic_value') else None
            ],
            'Undervaluation %': [
                results.get('undervaluation', None),
                results.get('undervaluation', None) + bull_adj if results.get('undervaluation') else None,
                results.get('undervaluation', None) + bear_adj if results.get('undervaluation') else None
            ]
        })
        st.dataframe(scenarios, use_container_width=True)
        
        st.header("Sensitivity Analysis (Heatmap)")
        heatmap = plot_heatmap(inputs, results.get('intrinsic_value'), inputs.get('wacc'), inputs.get('analyst_growth'))
        st.plotly_chart(heatmap, use_container_width=True)
        
        st.header("Monte Carlo Simulation")
        mc_results = run_monte_carlo(inputs, inputs.get('monte_carlo_runs', 1000), 
                                   inputs.get('growth_adj', 10.0), inputs.get('wacc_adj', 10.0))
        st.metric("Average Intrinsic Value", f"${mc_results.get('avg_value', 0):.2f}" if mc_results.get('avg_value') else "N/A")
        st.metric("Std Dev", f"${mc_results.get('std_dev', 0):.2f}" if mc_results.get('std_dev') else "N/A")
        st.metric("Probability Undervalued", f"{mc_results.get('prob_undervalued', 0):.2f}%" if mc_results.get('prob_undervalued') else "N/A")
        mc_plot = plot_monte_carlo(mc_results)
        st.plotly_chart(mc_plot, use_container_width=True)
        
        st.header("Model Comparison")
        model_comp = pd.DataFrame({
            'Model': ['Core', 'Lynch', 'DCF', 'DDM', 'Two-Stage DCF', 'RI', 'Reverse DCF', 'Graham'],
            'Intrinsic Value': [
                results.get('core_value', None), results.get('lynch_value', None), results.get('dcf_value', None),
                results.get('ddm_value', None), results.get('two_stage_dcf', None), results.get('ri_value', None),
                results.get('reverse_dcf_value', None), results.get('graham_value', None)
            ]
        })
        comp_plot = plot_model_comparison(model_comp)
        st.plotly_chart(comp_plot, use_container_width=True)

with tab2:
    st.header("Fundamental Graphs")
    ticker_graphs = st.text_input("Enter Ticker for Graphs", value=st.session_state.get('data', {}).get('ticker', ''))
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
st.markdown("*Disclaimer: This tool is for informational purposes only and not financial advice.*")
