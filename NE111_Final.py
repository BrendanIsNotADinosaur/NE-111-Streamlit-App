#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:37:28 2025

@author: bdong
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats


# ------------------ Helper functions ------------------ #

def parse_text_data(text: str):
    """Parse numbers from a textarea (commas, spaces, newlines, semicolons)."""
    if not text or not text.strip():
        return None

    tokens = re.split(r"[,\s;]+", text.strip())
    values = []
    for t in tokens:
        if not t:
            continue
        try:
            values.append(float(t))
        except ValueError:
            # Ignore non-numeric tokens
            pass
    if len(values) == 0:
        return None
    return np.array(values, dtype=float)


def compute_histogram(data, bins):
    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, edges, centers


def fit_distribution(dist, data):
    """Fit a scipy.stats distribution to data, return params and errors."""
    try:
        params = dist.fit(data)
    except Exception:
        return None, None, None

    # Histogram-based error metrics
    counts, edges, centers = compute_histogram(data, bins=30)
    try:
        pdf_vals = dist.pdf(centers, *params)
    except Exception:
        return params, None, None

    errors = pdf_vals - counts
    mse = float(np.mean(errors**2))
    max_err = float(np.max(np.abs(errors)))
    return params, mse, max_err


def format_params(dist, params):
    """Turn params into a nice readable dict: shape(s), loc, scale."""
    if params is None:
        return {}

    shapes = []
    if dist.shapes:
        shapes = [s.strip() for s in dist.shapes.split(",")]

    n_shapes = len(shapes)
    out = {}
    for i, name in enumerate(shapes):
        out[name] = params[i]

    out["loc"] = params[n_shapes]
    out["scale"] = params[n_shapes + 1]
    return out


def make_shape_slider(name, value, key):
    if not np.isfinite(value) or value == 0:
        min_val = 0.1
        max_val = 5.0
        value = 1.0
    else:
        factor = 4
        min_val = max(value / factor, 1e-6)
        max_val = max(value * factor, min_val * 2)
    return st.slider(
        f"Shape: {name}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(value),
        key=key,
    )


def make_loc_scale_sliders(dist_name, params, data, key_prefix=""):
    """Create sliders for loc and scale using data range."""
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    data_range = max(data_max - data_min, 1e-6)

    loc_val = params[-2]
    scale_val = params[-1]

    loc_min = data_min - 0.5 * data_range
    loc_max = data_max + 0.5 * data_range
    if not np.isfinite(loc_val):
        loc_val = 0.0
    loc_val = min(max(loc_val, loc_min), loc_max)

    scale_min = data_range / 100.0
    scale_max = data_range * 2.0
    if not np.isfinite(scale_val) or scale_val <= 0:
        scale_val = data_range / 10.0
    scale_val = min(max(scale_val, scale_min), scale_max)

    loc = st.slider(
        f"{dist_name}: loc",
        min_value=float(loc_min),
        max_value=float(loc_max),
        value=float(loc_val),
        key=key_prefix + "_loc",
    )
    scale = st.slider(
        f"{dist_name}: scale",
        min_value=float(scale_min),
        max_value=float(scale_max),
        value=float(scale_val),
        key=key_prefix + "_scale",
    )
    return loc, scale


# ------------------ Streamlit app ------------------ #

st.set_page_config(
    page_title="Histogram Distribution Fitter",
    layout="wide",
)

st.title("📊 Histogram Distribution Fitter")
st.markdown(
    """
Upload or type in data, fit several probability distributions, 
and explore both automatic and manual fitting.
"""
)

# Sidebar for data input
st.sidebar.header("1️⃣ Data input")

st.sidebar.markdown("**Option A:** Paste numbers (comma / space / newline separated)")
text_data = st.sidebar.text_area(
    "Typed data",
    placeholder="Example: 1.2, 1.3, 2.5, 2.6, 3.0, 3.1, ...",
    height=150,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Option B:** Upload CSV file")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

csv_column = None
csv_data = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            csv_column = st.sidebar.selectbox(
                "Select numeric column", numeric_cols, index=0
            )
            csv_data = df[csv_column].dropna().to_numpy(dtype=float)
        else:
            st.sidebar.warning("No numeric columns found in uploaded CSV.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

st.sidebar.markdown("---")
bins = st.sidebar.slider("Number of histogram bins", min_value=5, max_value=100, value=30)

# Decide which data to use
data = None
source = None

if csv_data is not None:
    data = csv_data
    source = f"CSV column: {csv_column}"
else:
    data = parse_text_data(text_data)
    if data is not None:
        source = "Typed data"

if data is None:
    st.info("👈 Enter data in the sidebar (typed or CSV) to get started.")
    st.stop()

# Basic info about data
st.success(f"Using **{len(data)}** data points from **{source}**.")

# Top layout: histogram + summary stats
col_hist, col_stats = st.columns([2, 1])

with col_hist:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=bins, density=True, alpha=0.4, edgecolor="black")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Data Histogram")
    st.pyplot(fig, use_container_width=True)

with col_stats:
    st.subheader("Summary statistics")
    st.write(pd.DataFrame(
        {
            "Statistic": ["Count", "Mean", "Std", "Min", "Max"],
            "Value": [
                len(data),
                np.mean(data),
                np.std(data, ddof=1),
                np.min(data),
                np.max(data),
            ],
        }
    ))

# Available distributions (>= 10)
available_distributions = {
    "Normal (norm)": stats.norm,
    "Exponential (expon)": stats.expon,
    "Gamma (gamma)": stats.gamma,
    "Lognormal (lognorm)": stats.lognorm,
    "Weibull (weibull_min)": stats.weibull_min,
    "Beta (beta)": stats.beta,
    "Chi-square (chi2)": stats.chi2,
    "Student t (t)": stats.t,
    "Uniform (uniform)": stats.uniform,
    "Pareto (pareto)": stats.pareto,
    "Triangular (triang)": stats.triang,
}

tab_auto, tab_manual = st.tabs(["⚙️ Automatic fitting", "🎛 Manual fitting"])

# ------------------ Automatic fitting ------------------ #
with tab_auto:
    st.subheader("Automatic distribution fitting")

    selected_names = st.multiselect(
        "Choose distributions to fit",
        list(available_distributions.keys()),
        default=["Normal (norm)", "Exponential (expon)", "Gamma (gamma)"],
    )

    if not selected_names:
        st.info("Select at least one distribution to fit.")
    else:
        # Create overlay plot
        x_min = float(np.min(data))
        x_max = float(np.max(data))
        x_range = x_max - x_min
        x_min_plot = x_min - 0.1 * x_range
        x_max_plot = x_max + 0.1 * x_range

        x = np.linspace(x_min_plot, x_max_plot, 400)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(data, bins=bins, density=True, alpha=0.3, edgecolor="black", label="Data")

        results = []

        for name in selected_names:
            dist = available_distributions[name]
            params, mse, max_err = fit_distribution(dist, data)
            if params is None:
                st.warning(f"Could not fit {name}.")
                continue

            try:
                y = dist.pdf(x, *params)
                ax2.plot(x, y, label=name)
            except Exception:
                st.warning(f"Could not compute PDF for {name}.")
                continue

            param_dict = format_params(dist, params)
            results.append(
                {
                    "Distribution": name,
                    "Parameters": ", ".join(
                        f"{k}={v:.4g}"
                        for k, v in param_dict.items()
                    ),
                    "MSE": mse if mse is not None else np.nan,
                    "Max error": max_err if max_err is not None else np.nan,
                }
            )

        ax2.set_xlabel("Value")
        ax2.set_ylabel("Density")
        ax2.set_title("Histogram with fitted distributions")
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

        if results:
            st.markdown("#### Fit quality")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("No successful fits to display.")

# ------------------ Manual fitting ------------------ #
with tab_manual:
    st.subheader("Manual fitting controls")

    manual_name = st.selectbox(
        "Choose distribution to manually adjust",
        list(available_distributions.keys()),
        index=0,
    )
    dist = available_distributions[manual_name]

    # Initial params from automatic fit (used as starting point)
    init_params, _, _ = fit_distribution(dist, data)
    if init_params is None:
        st.warning(f"Could not fit {manual_name} automatically. Using default params.")
        # Some generic defaults: shapes=(), loc=mean, scale=std
        init_params = (np.mean(data), np.std(data) if np.std(data) > 0 else 1.0)
        # For distributions without shapes, we'll fix below.

    shapes = []
    if dist.shapes:
        shapes = [s.strip() for s in dist.shapes.split(",")]
    n_shapes = len(shapes)

    # Ensure init_params length: shape params + loc + scale
    if len(init_params) < n_shapes + 2:
        # pad if needed
        extra = [1.0] * (n_shapes + 2 - len(init_params))
        init_params = tuple(list(init_params) + extra)

    # Sliders for shape parameters
    manual_params = []

    for i, shape_name in enumerate(shapes):
        shape_val = init_params[i]
        slider_val = make_shape_slider(shape_name, shape_val, key=f"{manual_name}_shape_{i}")
        manual_params.append(slider_val)

    # Sliders for loc and scale
    loc_val, scale_val = make_loc_scale_sliders(
        manual_name, init_params[-2:], data, key_prefix=manual_name
    )
    manual_params.extend([loc_val, scale_val])

    # Plot manual fit
    x_min = float(np.min(data))
    x_max = float(np.max(data))
    x_range = x_max - x_min
    x_min_plot = x_min - 0.1 * x_range
    x_max_plot = x_max + 0.1 * x_range

    x = np.linspace(x_min_plot, x_max_plot, 400)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    counts, edges, centers = compute_histogram(data, bins=bins)
    ax3.hist(data, bins=bins, density=True, alpha=0.3, edgecolor="black", label="Data")

    try:
        y = dist.pdf(x, *manual_params)
        y_centers = dist.pdf(centers, *manual_params)
        errors = y_centers - counts
        mse = float(np.mean(errors**2))
        max_err = float(np.max(np.abs(errors)))

        ax3.plot(x, y, linewidth=2.0, label=f"{manual_name} (manual)")
        ax3.set_xlabel("Value")
        ax3.set_ylabel("Density")
        ax3.set_title("Manual fit")
        ax3.legend()
        st.pyplot(fig3, use_container_width=True)

        param_dict = format_params(dist, tuple(manual_params))
        st.markdown("#### Manual parameters & fit quality")
        cols = st.columns(2)

        with cols[0]:
            st.write(pd.DataFrame(
                {
                    "Parameter": list(param_dict.keys()),
                    "Value": list(param_dict.values()),
                }
            ))

        with cols[1]:
            st.metric("MSE (hist vs PDF)", f"{mse:.4g}")
            st.metric("Max error", f"{max_err:.4g}")

    except Exception as e:
        st.error(f"Could not compute manual PDF: {e}")
        st.pyplot(fig3, use_container_width=True)
