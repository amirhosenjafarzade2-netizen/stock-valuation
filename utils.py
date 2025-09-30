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
    Validate all input parameters based on requirements from the HTML docs.
    Returns True if valid, False otherwise. More lenient for screener.
    """
    model = inputs.get('model')
    
    # Common validations
    if inputs.get('current_price') is None or inputs.get('current_price') <= 0:
        st.warning("Current Price must be positive. Skipping.")
        return False
    
    # Model-specific validations
    if model in ['Core Valuation (Excel)', 'Lynch Method', 'Residual Income (RI)', 'Graham Intrinsic Value']:
        if inputs.get('current_eps') is None or (model != 'Residual Income (RI)' and inputs.get('current_eps') == 0):  # Allow 0 for RI if handled
            st.warning(f"Current EPS must be provided and non-zero for {model} (except RI). Skipping.")
            return False
    if model in ['Core Valuation (Excel)', 'Lynch Method']:
        if inputs.get('forward_eps') is None or inputs.get('forward_eps') <= 0:
            st.warning(f"Forward EPS must be positive for {model}. Skipping.")
            return False
        if inputs.get('historical_pe') is None or inputs.get('historical_pe') <= 0:
            st.warning("Historical Avg P/E must be positive. Skipping.")
            return False
    if model in ['Core Valuation (Excel)']:
        if inputs.get('years_high_growth') is None or not (3 <= inputs.get('years_high_growth') <= 10):
            st.warning("Years (High-Growth) must be 3-10 years. Skipping.")
            return False
    if model == 'Dividend Discount Model (DDM)':
        if inputs.get('dividend_per_share') is None or inputs.get('dividend_per_share') <= 0:
            st.warning("Current Dividend Per Share must be positive for DDM. Skipping.")
            return False
    if model in ['Discounted Cash Flow (DCF)', 'Reverse DCF']:
        if (inputs.get('fcf') is None or inputs.get('fcf') == 0) and (inputs.get('current_eps') is None or inputs.get('current_eps') == 0):
            st.warning("Free Cash Flow or EPS must be provided and non-zero for DCF. Skipping.")
            return False
        if inputs.get('stable_growth') is None or inputs.get('wacc') is None or inputs.get('stable_growth') >= inputs.get('wacc'):
            st.warning("Stable Growth Rate must be < WACC. Skipping.")
            return False
    if model in ['Residual Income (RI)', 'Graham Intrinsic Value']:
        if inputs.get('book_value') is None or inputs.get('book_value') <= 0:
            st.warning("Book Value Per Share must be positive for RI/Graham. Skipping.")
            return False
    
    # Range validations (allow None for optional, but check if provided)
    ranges = {
        'desired_return': (0, 50),
        'analyst_growth': (-50, 50),  # Allowed negative
        'dividend_growth': (0, 50),
        'stable_growth': (0, 50),
        'tax_rate': (0, 100),
        'wacc': (0, 50),
        'roe': (-200, 200),  # Allowed more negative range
        'core_mos': (0, 100),
        'dividend_mos': (0, 100),
        'dcf_mos': (0, 100),
        'ri_mos': (0, 100),
        'exit_pe': (0, float('inf'))
    }
    
    for key, (min_val, max_val) in ranges.items():
        value = inputs.get(key)
        if value is not None and not (min_val <= value <= max_val):
            st.warning(f"{key.replace('_', ' ').title()} must be between {min_val} and {max_val}%. Skipping.")
            return False
    
    return True

def export_portfolio(portfolio_df, filename="portfolio.csv"):
    """
    Export portfolio DataFrame to CSV for download.
    """
    if portfolio_df.empty:
        st.warning("Portfolio is empty. Nothing to export.")
        return
    csv = portfolio_df.to_csv(index=False)
    st.download_button(
        label="Download Portfolio CSV",
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
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("Stock Valuation Report", title_style))
        story.append(Spacer(1, 12))
        
        # Valuation Results
        story.append(Paragraph("Valuation Results", styles['Heading2']))
        results_data = [
            ['Metric', 'Value'],
            ['Model', results.get('model', 'N/A')],
            ['Current Price', f"${results.get('current_price', 'N/A'):.2f}" if results.get('current_price') else 'N/A'],
            ['Intrinsic Value', f"${results.get('intrinsic_value', 'N/A'):.2f}" if results.get('intrinsic_value') else 'N/A'],
            ['Safe Buy Price', f"${results.get('safe_buy_price', 'N/A'):.2f}" if results.get('safe_buy_price') else 'N/A'],
            ['Undervaluation %', f"{results.get('undervaluation', 'N/A'):.2f}%" if results.get('undervaluation') else 'N/A'],
            ['Verdict', results.get('verdict', 'N/A')],
            ['PEG Ratio', f"{results.get('peg_ratio', 'N/A'):.2f}" if results.get('peg_ratio') else 'N/A'],
            ['Overall Score', f"{results.get('score', 'N/A')}/100" if results.get('score') else 'N/A']
        ]
        results_table = Table(results_data)
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(results_table)
        story.append(Spacer(1, 12))
        
        # Model Comparison
        story.append(Paragraph("Model Comparison", styles['Heading2']))
        model_data = [
            ['Model', 'Intrinsic Value'],
            ['Core', f"${results.get('core_value', 'N/A'):.2f}" if results.get('core_value') else 'N/A'],
            ['Lynch', f"${results.get('lynch_value', 'N/A'):.2f}" if results.get('lynch_value') else 'N/A'],
            ['DCF', f"${results.get('dcf_value', 'N/A'):.2f}" if results.get('dcf_value') else 'N/A'],
            ['DDM', f"${results.get('ddm_value', 'N/A'):.2f}" if results.get('ddm_value') else 'N/A'],
            ['Two-Stage DCF', f"${results.get('two_stage_dcf', 'N/A'):.2f}" if results.get('two_stage_dcf') else 'N/A'],
            ['RI', f"${results.get('ri_value', 'N/A'):.2f}" if results.get('ri_value') else 'N/A'],
            ['Reverse DCF', f"${results.get('reverse_dcf_value', 'N/A'):.2f}" if results.get('reverse_dcf_value') else 'N/A'],
            ['Graham', f"${results.get('graham_value', 'N/A'):.2f}" if results.get('graham_value') else 'N/A']
        ]
        model_table = Table(model_data)
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(model_table)
        story.append(Spacer(1, 12))
        
        # Portfolio Overview
        if not portfolio_df.empty:
            story.append(Paragraph("Portfolio Overview", styles['Heading2']))
            portfolio_data = [['Ticker', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta']]
            for _, row in portfolio_df.iterrows():
                portfolio_data.append([
                    row['Ticker'],
                    f"${row['Intrinsic Value']:.2f}" if pd.notna(row['Intrinsic Value']) else 'N/A',
                    f"{row['Undervaluation %']:.2f}%" if pd.notna(row['Undervaluation %']) else 'N/A',
                    row['Verdict'],
                    f"{row['Beta']:.2f}" if pd.notna(row['Beta']) else 'N/A'
                ])
            portfolio_table = Table(portfolio_data)
            portfolio_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(portfolio_table)
        
        # Disclaimer
        disclaimer = Paragraph(
            "Disclaimer: This tool is for informational purposes only and not financial advice. Verify all inputs and calculations independently.",
            styles['Normal']
        )
        story.append(Spacer(1, 12))
        story.append(disclaimer)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}. Ensure reportlab is installed and inputs are valid.")
        return None
