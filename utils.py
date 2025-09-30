import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

def validate_inputs(inputs):
    """
    Validate input parameters, allowing negative EPS/growth where applicable.
    """
    model = inputs['model']
    
    if inputs.get('current_price', 0) <= 0:
        st.warning("Current Price must be positive. Skipping.")
        return False
    
    if model in ['Core Valuation (Excel)', 'Lynch Method', 'Residual Income (RI)']:
        if inputs.get('current_eps') is None:
            st.warning(f"Current EPS must be provided for {model}. Skipping.")
            return False
    if model in ['Core Valuation (Excel)', 'Lynch Method']:
        if inputs.get('forward_eps', 0) <= 0:
            st.warning(f"Forward EPS must be positive for {model}. Skipping.")
            return False
        if inputs.get('historical_pe', 0) <= 0:
            st.warning("Historical Avg P/E must be positive. Skipping.")
            return False
    if model == 'Core Valuation (Excel)':
        if not (3 <= inputs.get('years_high_growth', 0) <= 10):
            st.warning("Years (High-Growth) must be 3-10 years. Skipping.")
            return False
    if model == 'Dividend Discount Model (DDM)':
        if inputs.get('dividend_per_share', 0) <= 0:
            st.warning("Current Dividend Per Share must be positive for DDM. Skipping.")
            return False
    if model in ['Discounted Cash Flow (DCF)', 'Reverse DCF']:
        if inputs.get('fcf') is None and inputs.get('current_eps') is None:
            st.warning("Free Cash Flow or EPS must be provided for DCF. Skipping.")
            return False
        if inputs.get('stable_growth', 0) >= inputs.get('wacc', 0):
            st.warning("Stable Growth Rate must be < WACC. Skipping.")
            return False
    if model in ['Residual Income (RI)', 'Graham Intrinsic Value']:
        if inputs.get('book_value', 0) <= 0:
            st.warning("Book Value Per Share must be positive for RI/Graham. Skipping.")
            return False
    
    ranges = {
        'desired_return': (0, 50),
        'analyst_growth': (-50, 50),  # CHANGED: Allow negative growth
        'dividend_growth': (0, 50),
        'stable_growth': (0, 50),
        'tax_rate': (0, 100),
        'wacc': (0, 50),
        'roe': (-200, 200),
        'core_mos': (0, 100),
        'dividend_mos': (0, 100),
        'dcf_mos': (0, 100),
        'ri_mos': (0, 100),
        'exit_pe': (0, float('inf'))
    }
    
    for key, (min_val, max_val) in ranges.items():
        if key in inputs and inputs[key] is not None and not (min_val <= inputs[key] <= max_val):
            st.warning(f"{key.replace('_', ' ').title()} must be {min_val}-{max_val}%. Skipping.")
            return False
    
    return True

def export_portfolio(portfolio_df, filename="portfolio.csv"):
    """
    Export portfolio DataFrame to CSV.
    """
    csv = portfolio_df.to_csv(index=False)
    st.download_button(
        label="Download Portfolio CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def download_csv(df, filename):
    """
    Helper function to download a DataFrame as CSV.
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def generate_pdf_report(results, portfolio_df, filename="valuation_report.pdf"):
    """
    Generate a PDF report with valuation results and portfolio overview.
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("Stock Valuation Report", title_style))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Valuation Results", styles['Heading2']))
        results_data = [
            ['Metric', 'Value'],
            ['Model', results.get('model', '-')],
            ['Current Price', f"${results.get('current_price', 0):.2f}"],
            ['Intrinsic Value', f"${results.get('intrinsic_value', 0):.2f}" if results.get('intrinsic_value') else "N/A"],
            ['Safe Buy Price', f"${results.get('safe_buy_price', 0):.2f}" if results.get('safe_buy_price') else "N/A"],
            ['Undervaluation %', f"{results.get('undervaluation', 0):.2f}%" if results.get('undervaluation') else "N/A"],
            ['Verdict', results.get('verdict', 'N/A')],
            ['PEG Ratio', f"{results.get('peg_ratio', 0):.2f}" if results.get('peg_ratio') else "N/A"],
            ['Overall Score', f"{results.get('score', 0)}/100" if results.get('score') else "N/A"]
        ]
        results_table = Table(results_data)
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-
