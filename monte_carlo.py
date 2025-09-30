import numpy as np
from valuation_models import calculate_valuation

def run_monte_carlo(inputs, num_runs=1000, growth_adj=10.0, wacc_adj=10.0):
    """
    Run Monte Carlo simulation by varying key inputs.
    """
    if num_runs <= 0 or inputs.get('current_price') is None:
        return {
            'values': np.array([]),
            'avg_value': None,
            'std_dev': None,
            'prob_undervalued': None,
            'num_runs': 0
        }
    
    model = inputs['model']
    base_growth = inputs.get('analyst_growth', 0)
    base_wacc = inputs.get('wacc', 8.0)
    
    growth_variations = np.clip(
        np.random.normal(base_growth, growth_adj, num_runs),
        max(-50, base_growth - 2 * growth_adj),  # CHANGED: Allow negative growth
        base_growth + 2 * growth_adj
    )
    wacc_variations = np.clip(
        np.random.normal(base_wacc, wacc_adj, num_runs),
        max(0.01, base_wacc - 2 * wacc_adj),
        base_wacc + 2 * wacc_adj
    )
    
    intrinsic_values = []
    valid_runs = 0
    for i in range(num_runs):
        sim_inputs = inputs.copy()
        sim_inputs['analyst_growth'] = growth_variations[i]
        sim_inputs['wacc'] = wacc_variations[i]
        
        if validate_sim_inputs(sim_inputs, model):
            sim_results = calculate_valuation(sim_inputs)
            value = sim_results.get('intrinsic_value', None)
            if value is not None:
                intrinsic_values.append(value)
                valid_runs += 1
    
    intrinsic_values = np.array(intrinsic_values)
    avg_value = np.nanmean(intrinsic_values) if len(intrinsic_values) > 0 else None
    std_dev = np.nanstd(intrinsic_values) if len(intrinsic_values) > 0 else None
    current_price = inputs['current_price']
    prob_undervalued = np.mean(intrinsic_values > current_price) * 100 if len(intrinsic_values) > 0 and current_price > 0 else None
    
    return {
        'values': intrinsic_values,
        'avg_value': avg_value,
        'std_dev': std_dev,
        'prob_undervalued': prob_undervalued,
        'num_runs': valid_runs
    }

def validate_sim_inputs(inputs, model):
    """
    Simplified validation for Monte Carlo runs.
    """
    if inputs.get('wacc', 0) <= 0:
        return False
    if model in ['Discounted Cash Flow (DCF)', 'Reverse DCF']:
        if inputs.get('stable_growth', 0) >= inputs.get('wacc', 0):
            return False
    return True
