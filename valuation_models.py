import numpy as np
import pandas as pd
import streamlit as st

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
    results['intrinsic_value'] = max(results.get('intrinsic_value', 0), 0)
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
    """
    FIXED: Now includes present value of dividends during growth period.
    Core valuation with proper discounting of interim cash flows.
    """
    years = inputs['years_high_growth']
    current_eps = inputs['current_eps']
    growth_rate = inputs['analyst_growth'] / 100
    exit_pe = inputs['exit_pe']
    discount_rate = inputs['desired_return'] / 100
    dividend_payout = inputs.get('dividend_per_share', 0) / max(current_eps, 0.01) if current_eps != 0 else 0
    dividend_payout = min(max(dividend_payout, 0), 1)  # Cap at 100%
    
    # Present value of dividends during growth period
    pv_dividends = 0
    for t in range(1, years + 1):
        eps_t = current_eps * (1 + growth_rate) ** t
        dividend_t = eps_t * dividend_payout
        pv_dividends += dividend_t / (1 + discount_rate) ** t
    
    # Terminal value
    future_eps = current_eps * (1 + growth_rate) ** years
    terminal_value = future_eps * exit_pe
    pv_terminal = terminal_value / (1 + discount_rate) ** years
    
    intrinsic_value = pv_dividends + pv_terminal
    return {'intrinsic_value': intrinsic_value}

def lynch_method(inputs):
    """Lynch Fair Value calculation - unchanged, this was correct."""
    current_eps = inputs['current_eps']
    growth_rate = inputs['analyst_growth']
    dividend_yield = (inputs.get('dividend_per_share', 0) / inputs['current_price'] * 100) if inputs['current_price'] > 0 else 0
    
    # Lynch uses growth + dividend yield for fair P/E
    fair_pe = growth_rate + dividend_yield
    intrinsic_value = current_eps * fair_pe
    return {'intrinsic_value': intrinsic_value}

def ddm_valuation(inputs):
    """Dividend Discount Model - unchanged, this was correct."""
    dividend = inputs['dividend_per_share']
    growth = inputs['dividend_growth'] / 100
    cost_equity = inputs['desired_return'] / 100
    
    if dividend <= 0 or cost_equity <= growth:
        return {'intrinsic_value': 0}
    intrinsic_value = dividend * (1 + growth) / (cost_equity - growth)
    return {'intrinsic_value': max(intrinsic_value, 0)}

def two_stage_dcf(inputs):
    """
    FIXED: No longer uses EPS as FCF fallback. Requires actual FCF.
    Two-Stage DCF with proper FCF requirement.
    """
    fcf = inputs.get('fcf', 0)
    
    # FIXED: No longer accept EPS as FCF
    if fcf <= 0:
        st.warning("DCF requires actual Free Cash Flow. EPS is not a substitute for FCF.")
        return {'intrinsic_value': 0}
    
    high_growth = inputs['analyst_growth'] / 100
    stable_growth = inputs['stable_growth'] / 100
    wacc = inputs['wacc'] / 100
    years_high = inputs['years_high_growth']
    
    # Validate stable growth < WACC
    if stable_growth >= wacc:
        return {'intrinsic_value': 0}
    
    # High growth phase
    pv_high = 0
    for t in range(1, years_high + 1):
        fcf_t = fcf * (1 + high_growth) ** t
        pv_high += fcf_t / (1 + wacc) ** t
    
    # Terminal value
    fcf_terminal = fcf * (1 + high_growth) ** years_high * (1 + stable_growth)
    terminal_value = fcf_terminal / (wacc - stable_growth)
    pv_terminal = terminal_value / (1 + wacc) ** years_high
    
    intrinsic_value = pv_high + pv_terminal
    return {'intrinsic_value': max(intrinsic_value, 0)}

def residual_income(inputs):
    """
    FIXED: Book value now grows by retained earnings, not growth rate.
    Residual Income model with proper equity dynamics.
    """
    book_value = inputs['book_value']
    roe = min(inputs['roe'] / 100, 1.0)
    cost_equity = inputs['desired_return'] / 100
    growth = inputs['analyst_growth'] / 100
    years = inputs['years_high_growth']
    
    # Calculate payout ratio from growth and ROE
    # Growth = ROE * (1 - payout_ratio)
    if roe > 0.001:
        retention_ratio = min(growth / roe, 1.0)
        payout_ratio = 1 - retention_ratio
    else:
        retention_ratio = 0
        payout_ratio = 1
    
    pv_ri = 0
    current_book = book_value
    
    for t in range(1, years + 1):
        # Earnings in year t
        earnings_t = current_book * roe
        
        # Residual income = Earnings - (Book Value * Cost of Equity)
        ri_t = earnings_t - (current_book * cost_equity)
        
        # Present value of RI
        pv_ri += ri_t / (1 + cost_equity) ** t
        
        # FIXED: Book value grows by retained earnings
        retained_earnings = earnings_t * retention_ratio
        current_book += retained_earnings
    
    # Terminal value
    if cost_equity > growth and roe > 0.001:
        terminal_earnings = current_book * roe
        terminal_ri = terminal_earnings - (current_book * cost_equity)
        # Perpetuity growth of RI
        terminal_value = terminal_ri * (1 + growth) / (cost_equity - growth)
        pv_terminal = terminal_value / (1 + cost_equity) ** years
    else:
        pv_terminal = 0
    
    intrinsic_value = book_value + pv_ri + pv_terminal
    return {'intrinsic_value': max(intrinsic_value, 0)}

def reverse_dcf(inputs):
    """
    FIXED: Now includes high-growth phase before terminal value.
    Reverse DCF to find implied growth rate.
    """
    current_price = inputs['current_price']
    fcf = inputs.get('fcf', 0)
    
    if fcf <= 0:
        st.warning("Reverse DCF requires actual Free Cash Flow.")
        return {'intrinsic_value': 0, 'implied_growth': 0}
    
    wacc = inputs['wacc'] / 100
    stable_growth = inputs['stable_growth'] / 100
    years_high = inputs['years_high_growth']
    
    def calculate_value(high_growth_rate):
        """Calculate firm value given a high growth rate."""
        g_high = min(high_growth_rate / 100, wacc - 0.001)
        
        # High growth phase
        pv_high = 0
        for t in range(1, years_high + 1):
            fcf_t = fcf * (1 + g_high) ** t
            pv_high += fcf_t / (1 + wacc) ** t
        
        # Terminal value
        if wacc > stable_growth:
            fcf_terminal = fcf * (1 + g_high) ** years_high * (1 + stable_growth)
            terminal_value = fcf_terminal / (wacc - stable_growth)
            pv_terminal = terminal_value / (1 + wacc) ** years_high
        else:
            pv_terminal = 0
        
        return pv_high + pv_terminal
    
    # Binary search for implied growth rate
    low, high = 0, min(wacc * 100 - 0.1, 50)
    implied_growth = 0
    
    for _ in range(100):
        mid = (low + high) / 2
        value = calculate_value(mid)
        
        if abs(value - current_price) < 0.01:
            implied_growth = mid
            break
        
        if value < current_price:
            low = mid
        else:
            high = mid
    
    implied_growth = (low + high) / 2
    
    # Calculate intrinsic value using analyst growth
    intrinsic_value = calculate_value(inputs['analyst_growth'])
    
    return {'intrinsic_value': max(intrinsic_value, 0), 'implied_growth': implied_growth}

def graham_intrinsic_value(inputs):
    """
    FIXED: Bond yield is now configurable, defaults to reasonable current rate.
    Graham Intrinsic Value formula.
    """
    eps = inputs['current_eps']
    book_value = inputs['book_value']
    growth = inputs['analyst_growth']
    
    # FIXED: Use configurable bond yield or reasonable default
    # Using 4.5% as a reasonable current AAA corporate bond yield
    aaa_bond_yield = inputs.get('aaa_bond_yield', 4.5)
    
    # Graham formula: V = EPS × (8.5 + 2g) × 4.4 / Y
    # where Y is current AAA corporate bond yield
    v = eps * (8.5 + 2 * growth) * 4.4 / aaa_bond_yield
    
    # Graham also suggested value shouldn't be less than 2/3 of book value
    intrinsic_value = max(v, 0.67 * book_value)
    
    return {'intrinsic_value': max(intrinsic_value, 0)}
