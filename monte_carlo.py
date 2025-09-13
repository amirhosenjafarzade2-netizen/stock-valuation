import numpy as np
from valuation_models import calculate_valuation

def run_monte_carlo(inputs, num_runs=1000, growth_adj=10.0, wacc_adj=10.0):
    """
    Run Monte Carlo simulation by varying key inputs (growth rate, WACC).
    Returns a dictionary with results including distribution of intrinsic values.
    """
    model = inputs['model']
    
    # Base values
    base_growth = inputs['analyst_growth']
    base_wacc = inputs['wacc']
    
    # Generate random variations (normal distribution, truncated)
    growth_variations = np.clip(
        np.random.normal(base_growth, growth_adj, num_runs),
        max(0, base_growth - 2 * growth_adj),
        base_growth + 2 * growth_adj
    )
    wacc_variations = np.clip(
        np.random.normal(base_wacc, wacc_adj, num_runs),
        max(0, base_wacc - 2 * wacc_adj),
        base_wacc + 2 * wacc_adj
    )
    
    # Run simulations
    intrinsic_values = []
    for i in range(num_runs):
        sim_inputs = inputs.copy()
        sim_inputs['analyst_growth'] = growth_variations[i]
        sim_inputs['wacc'] = wacc_variations[i]
        
        # Validate and calculate
        if validate_sim_inputs(sim_inputs, model):  # Pass model as argument
            sim_results = calculate_valuation(sim_inputs)
            intrinsic_values.append(sim_results.get('intrinsic_value', 0))
        else:
            intrinsic_values.append(0)  # Skip invalid runs
    
    # Calculate statistics
    intrinsic_values = np.array(intrinsic_values)
    avg_value = np.mean(intrinsic_values)
    std_dev = np.std(intrinsic_values)
    current_price = inputs['current_price']
    prob_undervalued = np.mean(intrinsic_values > current_price) * 100 if current_price > 0 else 0
    
    return {
        'values': intrinsic_values,
        'avg_value': avg_value,
        'std_dev': std_dev,
        'prob_undervalued': prob_undervalued,
        'num_runs': num_runs
    }

def validate_sim_inputs(inputs, model):
    """
    Simplified validation for Monte Carlo runs (faster than full validation).
    """
    if inputs['wacc'] <= 0 or inputs['analyst_growth'] < 0:
        return False
    if model in ['Discounted Cash Flow (DCF)', 'Reverse DCF']:
        if inputs['stable_growth'] >= inputs['wacc']:
            return False
    return True
