import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_heatmap(intrinsic_value, wacc, growth_rate):
    """
    Generate a heatmap for sensitivity analysis (intrinsic value vs WACC and growth rate).
    """
    # Define ranges for sensitivity analysis
    wacc_range = np.linspace(max(0.01, wacc - 2), min(50, wacc + 2), 10)  # ±2% around input WACC
    growth_range = np.linspace(max(0, growth_rate - 5), min(50, growth_rate + 5), 10)  # ±5% around growth
    
    # Calculate intrinsic values for each combination
    z = np.zeros((len(growth_range), len(wacc_range)))
    for i, g in enumerate(growth_range):
        for j, w in enumerate(wacc_range):
            # Simplified DCF for sensitivity (replace with actual model if needed)
            if w > g / 100:
                z[i, j] = intrinsic_value * (1 + g / 100) / (w / 100 - g / 100)
            else:
                z[i, j] = 0
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=wacc_range,
        y=growth_range,
        colorscale='Viridis',  # Color-blind friendly
        hovertemplate='WACC: %{x:.2f}%<br>Growth: %{y:.2f}%<br>Value: $%{z:.2f}<extra></extra>',
        zmin=np.min(z) * 0.8,
        zmax=np.max(z) * 1.2
    ))
    
    fig.update_layout(
        title='Sensitivity Analysis: Intrinsic Value vs WACC and Growth Rate',
        xaxis_title='WACC (%)',
        yaxis_title='Growth Rate (%)',
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )
    
    return fig

def plot_monte_carlo(mc_results):
    """
    Generate a histogram for Monte Carlo simulation results.
    """
    values = mc_results.get('values', np.random.normal(100, 20, 1000))  # Placeholder if empty
    avg_value = mc_results.get('avg_value', np.mean(values))
    std_dev = mc_results.get('std_dev', np.std(values))
    
    # Create histogram
    fig = px.histogram(
        x=values,
        nbins=50,
        title='Monte Carlo Simulation: Distribution of Intrinsic Values',
        labels={'x': 'Intrinsic Value ($)', 'y': 'Frequency'},
        template='plotly_white',
        color_discrete_sequence=['#636EFA']  # Color-blind friendly blue
    )
    
    # Add mean and std dev lines
    fig.add_vline(x=avg_value, line_dash="dash", line_color="green", annotation_text="Mean", annotation_position="top")
    fig.add_vline(x=avg_value - std_dev, line_dash="dot", line_color="orange", annotation_text="-1 SD", annotation_position="top left")
    fig.add_vline(x=avg_value + std_dev, line_dash="dot", line_color="orange", annotation_text="+1 SD", annotation_position="top right")
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )
    
    return fig

def plot_model_comparison(model_comp_df):
    """
    Generate a bar chart comparing intrinsic values across models.
    """
    fig = px.bar(
        model_comp_df,
        x='Model',
        y='Intrinsic Value',
        title='Model Comparison: Intrinsic Values',
        labels={'Intrinsic Value': 'Intrinsic Value ($)'},
        template='plotly_white',
        color='Model',
        color_discrete_sequence=px.colors.qualitative.Set2  # Color-blind friendly
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )
    
    return fig
