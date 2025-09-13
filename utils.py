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
    Returns True if valid, False otherwise.
    """
    model = inputs['model']
    
    # Common validations
    if inputs['current_price'] <= 0:
        st.error("Current Price must be positive.")
        return False
    
    # Model-specific validations
    if model in ['Core Valuation (Excel)', 'Lynch Method', 'Residual Income (RI)', 'Graham Intrinsic Value']:
        if inputs['current_eps'] == 0:
            st.error("Current EPS must be non-zero for Core/Lynch/RI/Graham.")
            return False
    if model in ['Core Valuation (Excel)', 'Lynch Method']:
        if inputs['forward_eps'] <= 0:
            st.error("Forward EPS must be positive for Core/Lynch.")
            return False
        if inputs['historical_pe'] <= 0:
            st.error("Historical Avg P/E must be positive.")
            return False
    if model in ['Core Valuation (Excel)']:
        if not (3 <= inputs['years_high_growth'] <= 10):
            st.error("Years (High-Growth) must be 3-10 years.")
            return False
    if model == 'Dividend Discount Model (DDM)':
        if inputs['dividend_per_share'] <= 0:
            st.error("Current Dividend Per Share must be positive for DDM.")
            return False
    if model in ['Discounted Cash Flow (DCF)', 'Reverse DCF']:
        if inputs['fcf'] == 0 and inputs['current_eps'] == 0:
            st.error("Free Cash Flow or EPS must be non-zero for DCF if no EPS.")
            return False
        if inputs['stable_growth'] >= inputs['wacc']:
            st.error("Stable Growth Rate must be < WACC.")
            return False
    if model in ['Residual Income (RI)', 'Graham Intrinsic Value']:
        if inputs['book_value'] <= 0:
            st.error("Book Value Per Share must be positive for RI/Graham.")
            return False
    
    # Range validations
    ranges = {
        'desired_return': (0, 50),
        'analyst_growth': (0, 50),
        'dividend_growth': (0, 50),
        'stable_growth': (0, 50),
        'tax_rate': (0, 100),
        'wacc': (0, 50),
        'roe': (0, 100),
        'core_mos': (0, 100),
        'dividend_mos': (0, 100),
        'dcf_mos': (0, 100),
        'ri_mos': (0, 100),
        'exit_pe': (0, float('inf'))
    }
    
    for key, (min_val, max_val) in ranges.items():
        if key in inputs and not (min_val <= inputs[key] <= max_val):
            st.error(f"{key.replace('_', ' ').title()} must be {min_val}-{max_val}%.")
            return False
    
    return True

def export_portfolio(portfolio_df, filename="portfolio.csv"):
    """
    Export portfolio DataFrame to CSV for download.
    """
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
    Returns bytes for Streamlit download.
    """
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
        ['Model', results.get('model', '-')],
        ['Current Price', f"${results.get('current_price', 0):.2f}"],
        ['Intrinsic Value', f"${results.get('intrinsic_value', 0):.2f}"],
        ['Safe Buy Price', f"${results.get('safe_buy_price', 0):.2f}"],
        ['Undervaluation %', f"{results.get('undervaluation', 0):.2f}%"],
        ['Verdict', results.get('verdict', '-')],
        ['PEG Ratio', f"{results.get('peg_ratio', 0):.2f}"],
        ['Overall Score', f"{results.get('score', 0)}/100"]
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
        ['Core', f"${results.get('core_value', 0):.2f}"],
        ['Lynch', f"${results.get('lynch_value', 0):.2f}"],
        ['DCF', f"${results.get('dcf_value', 0):.2f}"],
        ['DDM', f"${results.get('ddm_value', 0):.2f}"],
        ['Two-Stage DCF', f"${results.get('two_stage_dcf', 0):.2f}"],
        ['RI', f"${results.get('ri_value', 0):.2f}"],
        ['Reverse DCF', f"${results.get('reverse_dcf_value', 0):.2f}"],
        ['Graham', f"${results.get('graham_value', 0):.2f}"]
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
        portfolio_data = [['Ticker', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta']] + portfolio_df.values.tolist()
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
