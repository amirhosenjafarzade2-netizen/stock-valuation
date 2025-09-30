import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from valuation_models import two_stage_dcf
import streamlit as st

def plot_heatmap(inputs, intrinsic_value, wacc, growth_rate):
    """
    Generate a heatmap for sensitivity analysis using actual DCF.
    Args:
        inputs (dict): Dictionary of valuation inputs (e.g., from fetch_stock_data).
        intrinsic_value (float): Base intrinsic value.
        wacc (float): Weighted Average Cost of Capital (%).
        growth_rate (float): Analyst growth rate (%).
    Returns:
        go.Figure: Heatmap of intrinsic values.
    """
    if not inputs or intrinsic_value is None or wacc is None or growth_rate is None:
        st.warning("Cannot generate heatmap: Missing inputs, intrinsic value, WACC, or growth rate.")
        return go.Figure()

    # Flexible ranges based on input values
    wacc_range = np.linspace(max(0.01, wacc - 3), min(50, wacc + 3), 10)
    growth_range = np.linspace(max(-20, growth_rate - 10), min(50, growth_rate + 10), 10)

    z = np.zeros((len(growth_range), len(wacc_range)))
    for i, g in enumerate(growth_range):
        for j, w in enumerate(wacc_range):
            sim_inputs = inputs.copy()
            sim_inputs['analyst_growth'] = g
            sim_inputs['wacc'] = w
            try:
                results = two_stage_dcf(sim_inputs)
                z[i, j] = results.get('intrinsic_value', 0) or 0
            except Exception as e:
                st.warning(f"Error calculating DCF for WACC={w:.2f}%, Growth={g:.2f}%: {str(e)}")
                z[i, j] = 0

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=wacc_range,
        y=growth_range,
        colorscale='Viridis',
        hovertemplate='WACC: %{x:.2f}%<br>Growth: %{y:.2f}%<br>Value: $%{z:.2f}<extra></extra>',
        zmin=np.nanmin(z) * 0.8 if np.nanmin(z) != 0 else 0,
        zmax=np.nanmax(z) * 1.2 if np.nanmax(z) != 0 else 1
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
    values = mc_results.get('values', None)
    if values is None or not len(values):
        st.warning("Cannot generate Monte Carlo histogram: No simulation data available.")
        return go.Figure()

    avg_value = mc_results.get('avg_value', np.nanmean(values))
    std_dev = mc_results.get('std_dev', np.nanstd(values))

    fig = px.histogram(
        x=values,
        nbins=50,
        title='Monte Carlo Simulation: Distribution of Intrinsic Values',
        labels={'x': 'Intrinsic Value ($)', 'y': 'Frequency'},
        template='plotly_white',
        color_discrete_sequence=['#636EFA']
    )

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
    model_comp_df = model_comp_df.dropna(subset=['Intrinsic Value'])
    if model_comp_df.empty:
        st.warning("Cannot generate model comparison chart: No valid data available.")
        return go.Figure()

    fig = px.bar(
        model_comp_df,
        x='Model',
        y='Intrinsic Value',
        title='Model Comparison: Intrinsic Values',
        labels={'Intrinsic Value': 'Intrinsic Value ($)'},
        template='plotly_white',
        color='Model',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )

    return fig
