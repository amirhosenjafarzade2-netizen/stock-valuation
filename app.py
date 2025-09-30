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

# Initialize session state with type coercion
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Ticker', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta']).astype({
        'Ticker': 'str',
        'Intrinsic Value': 'float',
        'Undervaluation %': 'float',
        'Verdict': 'str',
        'Beta': 'float'
    })
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

# Load custom CSS
try:
    with open("styles.html") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("styles.html not found. Using default styling.")
    # Fallback inline CSS
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
            margin-bottom: 20px;
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

# Create tabs
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
        
        ticker = st.text_input("Ticker Symbol", help="Enter a valid ticker (e.g., AAPL).").upper()
        
        st.subheader("Custom Inputs (Optional)")
        current_price = st.number_input("Current Price", min_value=0.0, value=st.session_state.data.get('current_price', 0.0))
        current_eps = st.number_input("Current EPS", value=st.session_state.data.get('current_eps', 0.0))
        forward_eps = st.number_input("Forward EPS", value=st.session_state.data.get('forward_eps', 0.0))
        dividend_per_share = st.number_input("Current Dividend Per Share", min_value=0.0, value=st.session_state.data.get('dividend_per_share', 0.0))
        analyst_growth = st.number_input("Analyst Growth Rate %", min_value=-50.0, max_value=50.0, value=st.session_state.data.get('analyst_growth', 5.0))
        wacc = st.number_input("WACC %", min_value=0.0, max_value=50.0, value=st.session_state.data.get('wacc', 8.0))
        stable_growth = st.number_input("Stable Growth Rate %", min_value=0.0, max_value=50.0, value=st.session_state.data.get('stable_growth', 3.0))
        years_high_growth = st.number_input("Years of High Growth", min_value=3, max_value=10, value=st.session_state.data.get('years_high_growth', 5))
        historical_pe = st.number_input("Historical Avg P/E", min_value=0.0, value=st.session_state.data.get('historical_pe', 15.0))
        exit_pe = st.number_input("Exit P/E", min_value=0.0, value=st.session_state.data.get('exit_pe', 15.0))
        desired_return = st.number_input("Desired Return %", min_value=0.0, max_value=50.0, value=st.session_state.data.get('desired_return', 10.0))
        tax_rate = st.number_input("Tax Rate %", min_value=0.0, max_value=100.0, value=st.session_state.data.get('tax_rate', 25.0))
        core_mos = st.number_input("Core MOS %", min_value=0.0, max_value=100.0, value=st.session_state.data.get('core_mos', 25.0))
        dividend_mos = st.number_input("Dividend MOS %", min_value=0.0, max_value=100.0, value=st.session_state.data.get('dividend_mos', 25.0))
        dcf_mos = st.number_input("DCF MOS %", min_value=0.0, max_value=100.0, value=st.session_state.data.get('dcf_mos', 25.0))
        ri_mos = st.number_input("RI MOS %", min_value=0.0, max_value=100.0, value=st.session_state.data.get('ri_mos', 25.0))
        dividend_growth = st.number_input("Dividend Growth Rate %", min_value=0.0, max_value=50.0, value=st.session_state.data.get('dividend_growth', 5.0))
        fcf = st.number_input("Free Cash Flow", value=st.session_state.data.get('fcf', 0.0))
        beta = st.number_input("Beta", min_value=0.0, max_value=5.0, value=st.session_state.data.get('beta', 1.0))
        book_value = st.number_input("Book Value Per Share", min_value=0.0, value=st.session_state.data.get('book_value', 0.0))
        roe = st.number_input("ROE %", min_value=-100.0, max_value=100.0, value=st.session_state.data.get('roe', 0.0))
        monte_carlo_runs = st.number_input("Monte Carlo Runs", min_value=100, max_value=10000, value=st.session_state.data.get('monte_carlo_runs', 1000))
        growth_adj = st.number_input("Growth Adjustment %", min_value=0.0, max_value=50.0, value=st.session_state.data.get('growth_adj', 10.0))
        wacc_adj = st.number_input("WACC Adjustment %", min_value=0.0, max_value=50.0, value=st.session_state.data.get('wacc_adj', 10.0))
        
        st.subheader("Advanced Options")
        aaa_bond_yield = st.number_input("AAA Bond Yield %", min_value=0.0, max_value=10.0, value=4.8, help="For Graham model; fetch latest from FRED or use default.")
        
        if st.button("Fetch Stock Data"):
            with st.spinner("Fetching data from Yahoo Finance..."):
                try:
                    data = fetch_stock_data(ticker)
                    data.update({
                        'current_price': current_price or data.get('current_price'),
                        'current_eps': current_eps or data.get('current_eps'),
                        'forward_eps': forward_eps or data.get('forward_eps'),
                        'dividend_per_share': dividend_per_share or data.get('dividend_per_share'),
                        'analyst_growth': analyst_growth or data.get('analyst_growth'),
                        'wacc': wacc or data.get('wacc'),
                        'stable_growth': stable_growth or data.get('stable_growth'),
                        'years_high_growth': years_high_growth or data.get('years_high_growth'),
                        'historical_pe': historical_pe or data.get('historical_pe'),
                        'exit_pe': exit_pe or data.get('exit_pe'),
                        'desired_return': desired_return or data.get('desired_return'),
                        'tax_rate': tax_rate or data.get('tax_rate'),
                        'core_mos': core_mos or data.get('core_mos'),
                        'dividend_mos': dividend_mos or data.get('dividend_mos'),
                        'dcf_mos': dcf_mos or data.get('dcf_mos'),
                        'ri_mos': ri_mos or data.get('ri_mos'),
                        'dividend_growth': dividend_growth or data.get('dividend_growth'),
                        'fcf': fcf or data.get('fcf'),
                        'beta': beta or data.get('beta'),
                        'book_value': book_value or data.get('book_value'),
                        'roe': roe or data.get('roe'),
                        'monte_carlo_runs': monte_carlo_runs or data.get('monte_carlo_runs'),
                        'growth_adj': growth_adj or data.get('growth_adj'),
                        'wacc_adj': wacc_adj or data.get('wacc_adj'),
                        'aaa_bond_yield': aaa_bond_yield
                    })
                    st.session_state.data = data
                    st.success(f"Data fetched for {ticker}. Custom overrides applied.")
                except Exception as e:
                    st.error(f"Error fetching data for {ticker}: {str(e)}")
        
        if st.button("Calculate Valuation"):
            with st.spinner("Calculating valuation..."):
                inputs = st.session_state.data.copy()
                inputs['model'] = model
                if validate_inputs(inputs):
                    results = calculate_valuation(inputs)
                    st.session_state.results = results
                    st.success("Valuation calculated.")
                else:
                    st.error("Invalid inputs. Check warnings and adjust.")
        
        add_to_portfolio = st.checkbox("Add to Portfolio", value=False)
        export_report = st.checkbox("Export PDF Report", value=False)
    
    # Main content
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.header("Valuation Results")
        results = st.session_state.get('results', {})
        if results:
            st.metric("Current Price", f"${results.get('current_price', 'N/A'):.2f}" if results.get('current_price') else "N/A")
            st.metric("Intrinsic Value", f"${results.get('intrinsic_value', 'N/A'):.2f}" if results.get('intrinsic_value') else "N/A")
            st.metric("Safe Buy Price", f"${results.get('safe_buy_price', 'N/A'):.2f}" if results.get('safe_buy_price') else "N/A")
            st.metric("Undervaluation %", f"{results.get('undervaluation', 'N/A'):.2f}%" if results.get('undervaluation') else "N/A")
            st.metric("PEG Ratio", f"{results.get('peg_ratio', 'N/A'):.2f}" if results.get('peg_ratio') else "N/A")
            st.metric("PE Delta %", f"{results.get('pe_delta', 'N/A'):.2f}%" if results.get('pe_delta') else "N/A")
            st.metric("Required EPS CAGR %", f"{results.get('eps_cagr', 'N/A'):.2f}%" if results.get('eps_cagr') else "N/A")
            st.metric("Overall Score (0-100)", f"{results.get('score', 'N/A')}" if results.get('score') else "N/A")
            st.metric("Verdict", results.get('verdict', 'N/A'))
            st.metric("Lynch Fair Value", f"${results.get('lynch_value', 'N/A'):.2f}" if results.get('lynch_value') else "N/A")
            st.metric("DCF Intrinsic Value", f"${results.get('dcf_value', 'N/A'):.2f}" if results.get('dcf_value') else "N/A")
            st.metric("DDM Intrinsic Value", f"${results.get('ddm_value', 'N/A'):.2f}" if results.get('ddm_value') else "N/A")
            st.metric("Two-Stage DCF Value", f"${results.get('two_stage_dcf', 'N/A'):.2f}" if results.get('two_stage_dcf') else "N/A")
            st.metric("Residual Income Value", f"${results.get('ri_value', 'N/A'):.2f}" if results.get('ri_value') else "N/A")
            st.metric("Implied Growth Rate %", f"{results.get('implied_growth', 'N/A'):.2f}%" if results.get('implied_growth') else "N/A")
            st.metric("Graham Intrinsic Value", f"${results.get('graham_value', 'N/A'):.2f}" if results.get('graham_value') else "N/A")
        else:
            st.info("Calculate valuation to see results.")
    
    with col_right:
        st.header("Portfolio Overview")
        if not st.session_state.portfolio.empty:
            portfolio_beta = st.session_state.portfolio['Beta'].mean(skipna=True)
            expected_return = portfolio_beta * 8.0
            st.metric("Portfolio Beta", f"{portfolio_beta:.2f}" if not pd.isna(portfolio_beta) else "N/A")
            st.metric("Portfolio Expected Return", f"{expected_return:.2f}%" if not pd.isna(expected_return) else "N/A")
        else:
            st.metric("Portfolio Beta", "N/A")
            st.metric("Portfolio Expected Return", "N/A")
        
        if add_to_portfolio and 'results' in st.session_state and results:
            new_row = pd.DataFrame([{
                'Ticker': ticker,
                'Intrinsic Value': results.get('intrinsic_value'),
                'Undervaluation %': results.get('undervaluation'),
                'Verdict': results.get('verdict', 'N/A'),
                'Beta': beta
            }]).astype(st.session_state.portfolio.dtypes)
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.success(f"{ticker} added to portfolio.")
        
        st.dataframe(st.session_state.portfolio, use_container_width=True)
        
        if st.button("Export Portfolio CSV"):
            try:
                export_portfolio(st.session_state.portfolio, "portfolio.csv")
                st.success("Portfolio exported as portfolio.csv")
            except Exception as e:
                st.error(f"Error exporting portfolio: {str(e)}")
        
        if export_report and 'results' in st.session_state and results:
            try:
                pdf_data = generate_pdf_report(results, st.session_state.portfolio)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name="valuation_report.pdf",
                    mime="application/pdf"
                )
                st.success("PDF report generated.")
            except Exception as e:
                st.error(f"Error generating PDF report: {str(e)}")
        
        st.header("Scenario Analysis")
        with st.expander("Adjust Scenarios"):
            bull_adj = st.slider("Bull Case Adjustment (%)", -50.0, 50.0, 20.0, key="bull_adj")
            bear_adj = st.slider("Bear Case Adjustment (%)", -50.0, 50.0, -20.0, key="bear_adj")
        scenarios = pd.DataFrame({
            'Scenario': ['Base Case', 'Bull Case', 'Bear Case'],
            'Intrinsic Value': [
                results.get('intrinsic_value', 'N/A'),
                results.get('intrinsic_value', 0) * (1 + bull_adj/100) if results.get('intrinsic_value') else 'N/A',
                results.get('intrinsic_value', 0) * (1 + bear_adj/100) if results.get('intrinsic_value') else 'N/A'
            ],
            'Undervaluation %': [
                results.get('undervaluation', 'N/A'),
                results.get('undervaluation', 0) + bull_adj if results.get('undervaluation') else 'N/A',
                results.get('undervaluation', 0) + bear_adj if results.get('undervaluation') else 'N/A'
            ]
        })
        st.dataframe(scenarios, use_container_width=True)

        st.header("Sensitivity Analysis (Heatmap)")
        heatmap = plot_heatmap(results.get('intrinsic_value'), wacc, analyst_growth)
        st.plotly_chart(heatmap, use_container_width=True)

        st.header("Monte Carlo Simulation")
        mc_results = run_monte_carlo(inputs, monte_carlo_runs, growth_adj, wacc_adj)
        st.metric("Average Intrinsic Value", f"${mc_results.get('avg_value', 'N/A'):.2f}" if mc_results.get('avg_value') else "N/A")
        st.metric("Std Dev", f"${mc_results.get('std_dev', 'N/A'):.2f}" if mc_results.get('std_dev') else "N/A")
        st.metric("Probability Undervalued (> Current Price)", f"{mc_results.get('prob_undervalued', 'N/A'):.2f}%" if mc_results.get('prob_undervalued') else "N/A")
        mc_plot = plot_monte_carlo(mc_results)
        st.plotly_chart(mc_plot, use_container_width=True)

        st.header("Model Comparison")
        model_comp = pd.DataFrame({
            'Model': ['Core', 'Lynch', 'DCF', 'DDM', 'Two-Stage DCF', 'RI', 'Reverse DCF', 'Graham'],
            'Intrinsic Value': [
                results.get('core_value', 'N/A'), results.get('lynch_value', 'N/A'), results.get('dcf_value', 'N/A'),
                results.get('ddm_value', 'N/A'), results.get('two_stage_dcf', 'N/A'), results.get('ri_value', 'N/A'),
                results.get('reverse_dcf_value', 'N/A'), results.get('graham_value', 'N/A')
            ]
        })
        comp_plot = plot_model_comparison(model_comp)
        st.plotly_chart(comp_plot, use_container_width=True)

with tab2:
    st.header("Fundamental Graphs")
    ticker_graphs = st.text_input("Enter Ticker for Graphs", value=st.session_state.get('data', {}).get('ticker', ''), help="Enter a ticker (e.g., AAPL) to view fundamental graphs.").upper()
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
