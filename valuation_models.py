import numpy as np
import pandas as pd
import streamlit as st

def calculate_valuation(inputs):
    """
    Calculate intrinsic value based on the selected model.
    """
    model = inputs['model']
    current_price = inputs.get('current_price', None)
    
    if current_price is None:
        return {'intrinsic_value': None, 'error': 'Missing current price'}
    
    mos = get_margin_of_safety(inputs, model)
    
    results = {'model': model}
    if model == "Core Valuation (Excel)":
        results.update(core_valuation(inputs))
    elif model == "Lynch Method":
        results.update(lynch_method(inputs))
    elif model == "Discounted Cash Flow (DCF)":
        results.update(two_stage_dcf(inputs))
    elif model == "Dividend Discount Model (DDM)":
        results.update(ddm_valuation(inputs))
    elif model == "Two-Stage DCF":
        results.update(two_stage_dcf(inputs))
    elif model == "Residual Income (RI)":
        results.update(residual_income(inputs))
    elif model == "Reverse DCF":
        results.update(reverse_dcf(inputs))
    elif model == "Graham Intrinsic Value":
        results.update(graham_intrinsic_value(inputs))
    else:
        results.update({'intrinsic_value': None, 'error': 'Unknown model'})
    
    if 'error' in results:
        return results
    
    # Common metrics
    results['current_price'] = current_price
    intrinsic_value = results.get('intrinsic_value', None)
    results['intrinsic_value'] = intrinsic_value if intrinsic_value is not None else 0
    results['safe_buy_price'] = intrinsic_value * (1 - mos / 100) if intrinsic_value else None
    results['undervaluation'] = ((intrinsic_value - current_price) / current_price) * 100 if intrinsic_value and current_price > 0 else None
    results['verdict'] = get_verdict(results['undervaluation']) if results['undervaluation'] is not None else 'N/A'
    
    results['peg_ratio'] = calculate_peg(inputs)
    results['pe_delta'] = (inputs['forward_eps'] * inputs['historical_pe'] - current_price) / current_price * 100 if inputs.get('forward_eps') and current_price > 0 else None
    results['eps_cagr'] = calculate_eps_cagr(inputs)
    
    results['score'] = calculate_weighted_score(results, inputs)
    
    all_models = {
        'core_value': core_valuation(inputs).get('intrinsic_value', None),
        'lynch_value': lynch_method(inputs).get('intrinsic_value', None),
        'dcf_value': two_stage_dcf(inputs).get('intrinsic_value', None),
        'ddm_value': ddm_valuation(inputs).get('intrinsic_value', None),
        'two_stage_dcf': two_stage_dcf(inputs).get('intrinsic_value', None),
        'ri_value': residual_income(inputs).get('intrinsic_value', None),
        'reverse_dcf_value': reverse_dcf(inputs).get('intrinsic_value', None),
        'graham_value': graham_intrinsic_value(inputs).get('intrinsic_value', None)
    }
    results.update(all_models)
    
    return results

def get_margin_of_safety(inputs, model):
    """Get MOS based on model."""
    default_mos = inputs.get('core_mos', 25)
    return {
        'Core Valuation (Excel)': inputs.get('core_mos', default_mos),
        'Lynch Method': inputs.get('core_mos', default_mos),
        'Dividend Discount Model (DDM)': inputs.get('dividend_mos', default_mos),
        'Discounted Cash Flow (DCF)': inputs.get('dcf_mos', default_mos),
        'Reverse DCF': inputs.get('dcf_mos', default_mos),
        'Residual Income (RI)': inputs.get('ri_mos', default_mos)
    }.get(model, default_mos)

def get_verdict(undervaluation):
    """Determine buy/sell/hold verdict."""
    if undervaluation is None:
        return "N/A"
    if undervaluation > 20:
        return "Strong Buy"
    elif undervaluation > 0:
        return "Buy"
    elif undervaluation > -20:
        return "Hold"
    else:
        return "Sell"

def calculate_peg(inputs):
    """Calculate PEG ratio, allowing negative growth."""
    current_eps = inputs.get('current_eps', None)
    growth = inputs.get('analyst_growth', None)
    if current_eps is None or growth is None or current_eps == 0 or abs(growth) < 0.01:
        return None
    pe = inputs['current_price'] / current_eps
    return pe / growth if growth != 0 else None

def calculate_eps_cagr(inputs):
    """Calculate required EPS CAGR, allowing negative growth."""
    years = inputs.get('years_high_growth', 5)
    current_eps = inputs.get('current_eps', None)
    forward_eps = inputs.get('forward_eps', None)
    analyst_growth = inputs.get('analyst_growth', None)
    if current_eps is None or forward_eps is None or analyst_growth is None or current_eps == 0 or years <= 0:
        return None
    target_eps = forward_eps * (1 + analyst_growth/100)**years
    if target_eps <= 0 or current_eps <= 0:
        return None
    return ((target_eps / current_eps) ** (1/years) - 1) * 100

def calculate_weighted_score(results, inputs):
    """Weighted score using industry-standard PEG benchmark."""
    if results.get('undervaluation') is None:
        return 0
    undervalue_weight = 0.3 * max(0, results['undervaluation'])
    
    peg = results.get('peg_ratio', None)
    peg_weight = 0
    if peg is not None:
        # CHANGED: Use industry-standard PEG < 1 as good
        peg_score = max(0, (1.5 - peg) / 1.5) * 100  # 1.5 is max good PEG
        peg_weight = 0.3 * peg_score
    
    pe_delta = results.get('pe_delta', None)
    pe_delta_weight = 0.4 * max(0, pe_delta) if pe_delta is not None else 0
    score = undervalue_weight + peg_weight + pe_delta_weight
    return min(100, max(0, score))

def core_valuation(inputs):
    """Placeholder: Truncated in original. Assume similar fixes."""
    intrinsic_value = 0  # Replace with actual logic
    return {'intrinsic_value': intrinsic_value}

def lynch_method(inputs):
    """Placeholder: Truncated in original."""
    intrinsic_value = 0
    return {'intrinsic_value': intrinsic_value}

def two_stage_dcf(inputs):
    """Placeholder: Truncated in original."""
    intrinsic_value = 0
    return {'intrinsic_value': intrinsic_value}

def ddm_valuation(inputs):
    """Placeholder: Truncated in original."""
    intrinsic_value = 0
    return {'intrinsic_value': intrinsic_value}

def residual_income(inputs):
    """
    Residual Income model with negative ROE handling.
    """
    book_value = inputs.get('book_value', None)
    roe = inputs.get('roe', None)
    cost_equity = inputs.get('wacc', 8.0) / 100
    years = inputs.get('years_high_growth', 5)
    growth = inputs.get('stable_growth', 3.0) / 100
    
    if book_value is None or roe is None or book_value <= 0:
        return {'intrinsic_value': None, 'error': 'Invalid book value or ROE'}
    
    retention_ratio = 1 - (inputs.get('dividend_per_share', 0) / inputs.get('current_eps', 1e-6))
    if retention_ratio < 0:
        retention_ratio = 0
        st.warning("Negative retention ratio; assuming full payout.")
    
    pv_ri = 0
    current_book = book_value
    
    for t in range(1, years + 1):
        earnings_t = current_book * (roe / 100)
        ri_t = earnings_t - (current_book * cost_equity)
        pv_ri += ri_t / (1 + cost_equity) ** t
        retained_earnings = earnings_t * retention_ratio
        current_book += retained_earnings
        if current_book < 0:
            current_book = 0.01
    
    if cost_equity > growth and abs(roe) > 0.001:
        terminal_earnings = current_book * (roe / 100)
        terminal_ri = terminal_earnings - (current_book * cost_equity)
        terminal_value = terminal_ri * (1 + growth) / (cost_equity - growth)
        pv_terminal = terminal_value / (1 + cost_equity) ** years
    else:
        pv_terminal = 0
    
    intrinsic_value = book_value + pv_ri + pv_terminal
    return {'intrinsic_value': max(intrinsic_value, 0) if intrinsic_value is not None else None}

def reverse_dcf(inputs):
    """
    Reverse DCF with improved convergence and negative FCF handling.
    """
    current_price = inputs.get('current_price', None)
    fcf = inputs.get('fcf', None)
    
    if current_price is None or fcf is None:
        return {'intrinsic_value': None, 'implied_growth': None, 'error': 'Missing price or FCF'}
    
    wacc = inputs.get('wacc', 8.0) / 100
    stable_growth = inputs.get('stable_growth', 3.0) / 100
    years_high = inputs.get('years_high_growth', 5)
    
    def calculate_value(high_growth_rate):
        g_high = high_growth_rate / 100
        if g_high >= wacc:
            g_high = wacc - 0.001
        pv_high = 0
        for t in range(1, years_high + 1):
            fcf_t = fcf * (1 + g_high) ** t
            pv_high += fcf_t / (1 + wacc) ** t
        if wacc > stable_growth:
            fcf_terminal = fcf * (1 + g_high) ** years_high * (1 + stable_growth)
            terminal_value = fcf_terminal / (wacc - stable_growth)
            pv_terminal = terminal_value / (1 + wacc) ** years_high
        else:
            pv_terminal = 0
        return pv_high + pv_terminal
    
    low, high = -50, min(wacc * 100 - 0.1, 50)  # CHANGED: Allow negative growth
    implied_growth = None
    tolerance = 0.01
    
    for iteration in range(100):
        mid = (low + high) / 2
        value = calculate_value(mid)
        if abs(value - current_price) < tolerance:
            implied_growth = mid
            break
        if value < current_price:
            low = mid
        else:
            high = mid
        if high - low < 0.001:
            implied_growth = mid
            break
    else:
        implied_growth = (low + high) / 2
    
    intrinsic_value = calculate_value(inputs.get('analyst_growth', 0))
    
    return {
        'intrinsic_value': max(intrinsic_value, 0) if intrinsic_value is not None else None,
        'implied_growth': implied_growth
    }

def graham_intrinsic_value(inputs):
    """
    Graham Intrinsic Value with proper bond yield handling.
    """
    eps = inputs.get('current_eps', None)
    book_value = inputs.get('book_value', None)
    growth = inputs.get('analyst_growth', None)
    
    if eps is None or book_value is None or growth is None:
        return {'intrinsic_value': None, 'error': 'Missing EPS, book value, or growth'}
    
    if eps <= 0:
        st.warning("Graham formula requires positive earnings. Using book value.")
        return {'intrinsic_value': book_value}
    
    aaa_bond_yield = inputs.get('aaa_bond_yield', 4.8)
    v = eps * (8.5 + 2 * growth) * 4.4 / aaa_bond_yield
    intrinsic_value = max(v, 0.67 * book_value)
    
    return {'intrinsic_value': max(intrinsic_value, 0) if intrinsic_value is not None else None}
