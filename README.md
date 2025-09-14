Stock Valuation Dashboard
A comprehensive Streamlit-based web application for stock valuation analysis, combining multiple financial models (Core, Lynch, DCF, DDM, Two-Stage DCF, Residual Income, Reverse DCF, Graham Intrinsic Value). Fetch real-time data from Yahoo Finance, perform Monte Carlo simulations, sensitivity analysis, and manage a personal portfolio. Ideal for individual investors and traders seeking actionable insights.
 
Features

Valuation Models: All 8 models with detailed calculations and comparisons.
Data Integration: Auto-fetch from Yahoo Finance (e.g., EPS, dividends, beta).
Portfolio Management: Add/remove stocks, calculate beta and expected return.
Advanced Analytics: Monte Carlo simulations, sensitivity heatmaps, scenario analysis.
Exports: CSV portfolios and PDF reports with charts.
UI: Responsive design with dark/light mode, collapsible inputs, interactive Plotly charts.

Prerequisites

Python 3.8+
Git (for cloning the repo)

Installation

Clone the repository:
git clone https://github.com/yourusername/stock-valuation-dashboard.git
cd stock-valuation-dashboard


Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt



Usage

Run the app:
streamlit run app.py

This opens the dashboard in your browser (default: http://localhost:8501).

Quick Start:

Enter a ticker (e.g., AAPL) and click Fetch to auto-populate data.
Select a valuation model from the dropdown.
Adjust inputs in the sidebar expanders (Core, Dividend, DCF, etc.).
Click Calculate to run analysis.
Add to Portfolio for multi-stock tracking.
Use Export Portfolio or Download Report for outputs.


Key Sections:

Valuation Dashboard: Core metrics (intrinsic value, undervaluation %, verdict).
Portfolio Overview: Weighted beta, expected return, editable table.
Scenario Analysis: Base/Bull/Bear cases.
Sensitivity Analysis: Heatmap of value vs. WACC/growth.
Monte Carlo Simulation: Probability of undervaluation with histogram.
Model Comparison: Bar chart across all models.



Project Structure
stock-valuation-dashboard/
├── app.py                 # Main Streamlit app
├── valuation_models.py    # All valuation calculations
├── data_fetch.py          # Yahoo Finance integration
├── visualizations.py      # Plotly charts (heatmaps, histograms)
├── utils.py               # Validation, exports (CSV/PDF)
├── monte_carlo.py         # Simulation logic
├── styles.html            # Custom CSS
├── requirements.txt       # Dependencies
├── README.md              # This file
└── .gitignore             # Git ignores

Deployment to Streamlit Cloud

Push to GitHub: Create a repo and push your code.
Go to share.streamlit.io.
Connect your GitHub repo and deploy (select app.py as entry point).
Access your public app URL instantly!

Contributing

Fork the repo and create a pull request.
Add new models or features via issues/PRs.
Ensure code follows PEP 8 and tests pass.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Disclaimer
This tool is for informational and educational purposes only. It is not financial advice. Always verify calculations and conduct your own research before making investment decisions. Past performance does not guarantee future results.
Support

Report bugs: Open an issue on GitHub.
Questions: Check the code or extend via PRs.


Built with ❤️ for personal investing. Contributions welcome!
