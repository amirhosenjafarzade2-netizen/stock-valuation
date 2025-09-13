import numpy as np
import pandas as pd

def calculate_valuation(inputs):
    """
    Main function to calculate intrinsic value based on the selected model.
    Returns a dictionary with all relevant metrics.
    """
    model = inputs['model']
    current_price = inputs['current_price']
    
    # Apply margin of safety
    mos = get_margin_of_safety(inputs, model)
    
    if model == "Core Valuation (Excel)":
        results = core_valuation(inputs)
    elif model == "Lynch Method":
        results = lynch_method(inputs)
    elif model == "Discounted Cash Flow (DCF)":
        results = dcf_valuation(inputs)
    elif model == "Dividend Discount Model (DDM)":
        results = ddm_valuation(inputs)
    elif model == "Two-Stage DCF":
        results = two_stage_dcf(inputs)
    elif model == "Residual Income (RI)":
        results = residual_income(inputs)
    elif model == "Reverse DCF":
        results = reverse_dcf(inputs)
    elif model == "Graham Intrinsic Value":
        results = graham_intrinsic_value(inputs)
    else:
        results = {'intrinsic_value': 0, 'error': 'Unknown model'}
    
    # Common metrics
    results['current_price'] = current_price
    results['intrinsic_value'] = results.get('intrinsic_value', 0) * (1 - mos / 100)
    results['safe_buy_price'] = results['intrinsic_value'] * (1 - mos / 100)
    results['undervaluation'] = ((results['intrinsic_value'] - current_price) / current_price) * 100 if current_price > 0 else 0
    results['verdict'] = get_verdict(results['undervaluation'])
    results['score'] = min(100, max(0, results['undervaluation'] + 50))  # Simple scoring
    
    # Model-specific metrics
    results['peg_ratio'] = calculate_peg(inputs)
    results['pe_delta'] = (inputs['forward_eps'] * inputs['historical_pe'] - current_price) / current_price * 100 if inputs['forward_eps'] > 0 else 0
    results['eps_cagr'] = calculate_eps_cagr(inputs)
    
    # Calculate all model values for comparison
    all_models = {
        'core_value': core_valuation(inputs).get('intrinsic_value', 0),
        'lynch_value': lynch_method(inputs).get('intrinsic_value', 0),
        'dcf_value': dcf_valuation(inputs).get('intrinsic_value', 0),
        'ddm_value': ddm_valuation(inputs).get('intrinsic_value', 0),
        'two_stage_dcf': two_stage_dcf(inputs).get('intrinsic_value', 0),
        'ri_value': residual_income(inputs).get('intrinsic_value', 0),
        'reverse_dcf_value': reverse_dcf(inputs).get('intrinsic_value', 0),
        'graham_value': graham_intrinsic_value(inputs).get('intrinsic_value', 0)
    }
    results.update(all_models)
    
    return results

def get_margin_of_safety(inputs, model):
    """Get MOS based on model."""
    if model in ['Core Valuation (Excel)', 'Lynch Method']:
        return inputs['core_mos']
    elif model == 'Dividend Discount Model (DDM)':
        return inputs['dividend_mos']
    elif model in ['Discounted Cash Flow (DCF)', 'Reverse DCF']:
        return inputs['dcf_mos']
    elif model == 'Residual Income (RI)':
        return inputs['ri_mos']
    else:
        return inputs.get('core_mos', 25)

def get_verdict(undervaluation):
    """Determine buy/sell/hold verdict."""
    if undervaluation > 20:
        return "Strong Buy"
    elif undervaluation > 0:
        return "Buy"
    elif undervaluation > -20:
        return "Hold"
    else:
        return "Sell"

def calculate_peg(inputs):
    """Calculate PEG ratio: P/E divided by growth rate."""
    pe = inputs['current_price'] / inputs['current_eps'] if inputs['current_eps'] != 0 else 0
    growth = inputs['analyst_growth']
    return pe / growth if growth > 0 else 0

def calculate_eps_cagr(inputs):
    """Calculate required EPS CAGR."""
    years = inputs['years_high_growth']
    current_eps = inputs['current_eps']
    target_eps = inputs['forward_eps'] * (1 + inputs['analyst_growth']/100)**years
    if current_eps > 0 and years > 0:
        return ((target_eps / current_eps) ** (1/years) - 1) * 100
    return 0

# Core Valuation (Excel) - P/E based projection
def core_valuation(inputs):
    years = inputs['years_high_growth']
    current_eps = inputs['current_eps']
    growth_rate = inputs['analyst_growth'] / 100
    exit_pe = inputs['exit_pe']
    
    # Project future EPS
    future_eps = current_eps * (1 + growth_rate) ** years
    
    # Terminal value
    terminal_value = future_eps * exit_pe
    
    # Discount back (simplified, assuming constant discount rate)
    discount_rate = inputs['desired_return'] / 100
    pv_terminal = terminal_value / (1 + discount_rate) ** years
    
    intrinsic_value = pv_terminal  # Simplified core model
    return {'intrinsic_value': intrinsic_value}

# Lynch Method - Earnings growth focused
def lynch_method(inputs):
    current_eps = inputs['current_eps']
    growth_rate = inputs['analyst_growth']
    pe = inputs['historical_pe']
    
    # Lynch fair P/E = growth rate
    fair_pe = growth_rate
    intrinsic_value = current_eps * fair_pe
    return {'intrinsic_value': intrinsic_value}

# DCF Valuation - Single stage
def dcf_valuation(inputs):
    fcf = inputs['fcf'] or inputs['current_eps'] * (1 - inputs['tax_rate']/100)  # FCF proxy if not provided
    wacc = inputs['wacc'] / 100
    stable_growth = inputs['stable_growth'] / 100
    
    # Terminal value
    terminal_value = fcf * (1 + stable_growth) / (wacc - stable_growth)
    
    # Simplified PV (assuming perpetual from now)
    intrinsic_value = terminal_value / (1 + wacc)  # One-year discount for simplicity
    return {'intrinsic_value': intrinsic_value}

# DDM - Gordon Growth Model
def ddm_valuation(inputs):
    dividend = inputs['dividend_per_share']
    growth = inputs['dividend_growth'] / 100
    cost_equity = inputs['desired_return'] / 100
    
    if cost_equity > growth:
        intrinsic_value = dividend * (1 + growth) / (cost_equity - growth)
    else:
        intrinsic_value = 0  # Invalid
    return {'intrinsic_value': intrinsic_value}

# Two-Stage DCF
def two_stage_dcf(inputs):
    fcf = inputs['fcf'] or inputs['current_eps']
    high_growth = inputs['analyst_growth'] / 100
    stable_growth = inputs['stable_growth'] / 100
    wacc = inputs['wacc'] / 100
    years_high = inputs['years_high_growth']
    
    # High growth period
    pv_high = 0
    for t in range(1, years_high + 1):
        fcf_t = fcf * (1 + high_growth) ** t
        pv_high += fcf_t / (1 + wacc) ** t
    
    # Terminal value
    fcf_terminal = fcf * (1 + high_growth) ** years_high * (1 + stable_growth)
    terminal_value = fcf_terminal / (wacc - stable_growth)
    pv_terminal = terminal_value / (1 + wacc) ** years_high
    
    intrinsic_value = pv_high + pv_terminal
    return {'intrinsic_value': intrinsic_value}

# Residual Income
def residual_income(inputs):
    book_value = inputs['book_value']
    roe = inputs['roe'] / 100
    cost_equity = inputs['desired_return'] / 100
    growth = inputs['analyst_growth'] / 100
    years = inputs['years_high_growth']
    
    # RI for high growth
    pv_ri = 0
    current_earnings = book_value * roe
    for t in range(1, years + 1):
        ri_t = current_earnings * growth ** (t-1) - book_value * cost_equity * (1 + growth) ** (t-1)
        pv_ri += ri_t / (1 + cost_equity) ** t
    
    # Terminal RI (perpetual)
    terminal_ri = (current_earnings * growth ** years - book_value * (1 + growth) ** years * cost_equity) / cost_equity
    pv_terminal = terminal_ri / (1 + cost_equity) ** years
    
    intrinsic_value = book_value + pv_ri + pv_terminal
    return {'intrinsic_value': intrinsic_value}

# Reverse DCF - Implied growth rate
def reverse_dcf(inputs):
    current_price = inputs['current_price']
    fcf = inputs['fcf'] or inputs['current_eps']
    wacc = inputs['wacc'] / 100
    years = inputs['years_high_growth']
    
    # Solve for implied growth where PV = current price
    # Simplified: assume single stage for reverse
    def pv_func(g):
        terminal = fcf * (1 + g) / (wacc - g)
        return terminal / (1 + wacc)
    
    # Binary search for g
    low, high = 0, 0.5
    for _ in range(50):  # Precision
        mid = (low + high) / 2
        if pv_func(mid) < current_price:
            low = mid
        else:
            high = mid
    implied_growth = low * 100
    
    # Forward value using analyst growth instead
    intrinsic_value = pv_func(inputs['analyst_growth'] / 100)
    return {'intrinsic_value': intrinsic_value, 'implied_growth': implied_growth}

# Graham Intrinsic Value
def graham_intrinsic_value(inputs):
    eps = inputs['current_eps']
    book_value = inputs['book_value']
    growth = inputs['analyst_growth']
    
    # Graham formula: V = EPS * (8.5 + 2g) * 4.4 / Y (simplified, assuming Y=4.4)
    v = eps * (8.5 + 2 * growth) * 4.4 / 4.4
    intrinsic_value = max(v, 0.67 * book_value)  # Take higher of earnings or asset value
    return {'intrinsic_value': intrinsic_value}
