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
        results = two_stage_dcf(inputs)
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
    results['intrinsic_value'] = max(results.get('intrinsic_value', 0), 0) * (1 - mos / 100)
    results['safe_buy_price'] = results['intrinsic_value'] * (1 - mos / 100)
    results['undervaluation'] = ((results['intrinsic_value'] - current_price) / current_price) * 100 if current_price > 0 else 0
    results['verdict'] = get_verdict(results['undervaluation'])
    
    # Model-specific metrics
    results['peg_ratio'] = calculate_peg(inputs)
    results['pe_delta'] = (inputs['forward_eps'] * inputs['historical_pe'] - current_price) / current_price * 100 if inputs['forward_eps'] > 0 else 0
    results['eps_cagr'] = calculate_eps_cagr(inputs)
    
    # Calculate weighted score after setting all metrics
    results['score'] = calculate_weighted_score(results, inputs)
    
    # Calculate all model values for comparison
    all_models = {
        'core_value': core_valuation(inputs).get('intrinsic_value', 0),
        'lynch_value': lynch_method(inputs).get('intrinsic_value', 0),
        'dcf_value': two_stage_dcf(inputs).get('intrinsic_value', 0),
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
    return pe / growth if growth > 0.01 else 0

def calculate_eps_cagr(inputs):
    """Calculate required EPS CAGR."""
    years = inputs['years_high_growth']
    current_eps = inputs['current_eps']
    target_eps = inputs['forward_eps'] * (1 + inputs['analyst_growth']/100)**years
    if current_eps != 0 and years > 0:
        return ((target_eps / current_eps) ** (1/years) - 1) * 100
    return 0

def calculate_weighted_score(results, inputs):
    """Weighted score: 30% undervaluation, 30% PEG (inverted), 40% P/E delta."""
    undervalue_weight = 0.3 * max(0, results['undervaluation'])
    peg_weight = 0.3 * (1 / (results['peg_ratio'] + 1)) * 100
    pe_delta_weight = 0.4 * max(0, results['pe_delta'])
    return min(100, max(0, undervalue_weight + peg_weight + pe_delta_weight))

def core_valuation(inputs):
    years = inputs['years_high_growth']
    current_eps = inputs['current_eps']
    growth_rate = inputs['analyst_growth'] / 100
    exit_pe = inputs['exit_pe']
    
    future_eps = current_eps * (1 + growth_rate) ** years
    terminal_value = future_eps * exit_pe
    discount_rate = inputs['desired_return'] / 100
    pv_terminal = terminal_value / (1 + discount_rate) ** years
    
    intrinsic_value = pv_terminal
    return {'intrinsic_value': intrinsic_value}

def lynch_method(inputs):
    current_eps = inputs['current_eps']
    growth_rate = inputs['analyst_growth']
    pe = inputs['historical_pe']
    
    fair_pe = growth_rate * 1.5
    intrinsic_value = current_eps * fair_pe
    return {'intrinsic_value': intrinsic_value}

def dcf_valuation(inputs):
    return two_stage_dcf(inputs)

def ddm_valuation(inputs):
    dividend = inputs['dividend_per_share']
    growth = inputs['dividend_growth'] / 100
    cost_equity = inputs['desired_return'] / 100
    
    if dividend <= 0 or cost_equity <= growth:
        return {'intrinsic_value': 0}
    intrinsic_value = dividend * (1 + growth) / (cost_equity - growth)
    return {'intrinsic_value': max(intrinsic_value, 0)}

def two_stage_dcf(inputs):
    fcf = inputs['fcf'] or inputs['current_eps'] * (1 - inputs['tax_rate']/100)
    high_growth = inputs['analyst_growth'] / 100
    stable_growth = inputs['stable_growth'] / 100
    wacc = inputs['wacc'] / 100
    years_high = inputs['years_high_growth']
    
    pv_high = 0
    for t in range(1, years_high + 1):
        fcf_t = fcf * (1 + high_growth) ** t
        pv_high += fcf_t / (1 + wacc) ** t
    
    if wacc > stable_growth:
        fcf_terminal = fcf * (1 + high_growth) ** years_high * (1 + stable_growth)
        terminal_value = fcf_terminal / (wacc - stable_growth)
        pv_terminal = terminal_value / (1 + wacc) ** years_high
    else:
        pv_terminal = 0
    
    intrinsic_value = pv_high + pv_terminal
    return {'intrinsic_value': max(intrinsic_value, 0)}

def residual_income(inputs):
    book_value = inputs['book_value']
    roe = min(inputs['roe'] / 100, 1.0)
    cost_equity = inputs['desired_return'] / 100
    growth = inputs['analyst_growth'] / 100
    years = inputs['years_high_growth']
    
    pv_ri = 0
    current_earnings = book_value * roe
    for t in range(1, years + 1):
        earnings_t = current_earnings * (1 + growth) ** (t - 1)
        book_t = book_value * (1 + growth) ** (t - 1)
        ri_t = earnings_t - book_t * cost_equity
        pv_ri += ri_t / (1 + cost_equity) ** t
    
    if cost_equity > growth:
        terminal_ri = (current_earnings * (1 + growth) ** years - book_value * (1 + growth) ** years * cost_equity) * growth / (cost_equity - growth)
        pv_terminal = terminal_ri / (1 + cost_equity) ** years
    else:
        pv_terminal = 0
    
    intrinsic_value = book_value + pv_ri + pv_terminal
    return {'intrinsic_value': max(intrinsic_value, 0)}

def reverse_dcf(inputs):
    current_price = inputs['current_price']
    fcf = inputs['fcf'] or inputs['current_eps']
    wacc = inputs['wacc'] / 100
    years = inputs['years_high_growth']
    
    def pv_func(g):
        g_rate = min(g / 100, wacc - 0.01)
        terminal = fcf * (1 + g_rate) / (wacc - g_rate) if wacc > g_rate else 0
        return terminal / (1 + wacc)
    
    low, high = 0, 100
    for _ in range(50):
        mid = (low + high) / 2
        if pv_func(mid) < current_price:
            low = mid
        else:
            high = mid
    implied_growth = low
    
    intrinsic_value = pv_func(inputs['analyst_growth'])
    return {'intrinsic_value': max(intrinsic_value, 0), 'implied_growth': implied_growth}

def graham_intrinsic_value(inputs):
    eps = inputs['current_eps']
    book_value = inputs['book_value']
    growth = inputs['analyst_growth']
    
    v = eps * (8.5 + 2 * growth) * 4.4 / 4.4
    intrinsic_value = max(v, 0.67 * book_value)
    return {'intrinsic_value': max(intrinsic_value, 0)}
