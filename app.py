import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="S&P 500 Strategy Dashboard")

# --- 2. DATA LOADING ---
@st.cache_data
def load_data(file_source):
    # Header is in the second row (index 1)
    df = pd.read_excel(file_source, header=1)
    # Filter to essential columns
    df = df[['Year', 'Total return']].dropna()
    df['Year'] = df['Year'].astype(int)
    # Sort ascending for chronological processing
    df = df.sort_values('Year').reset_index(drop=True)
    # Decimal returns
    df['ret_decimal'] = df['Total return'] / 100.0
    return df

# Initialize Data
DEFAULT_FILE = "SP_total_return.xlsx"
uploaded_file = st.sidebar.file_uploader("Upload S&P 500 Data", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
elif os.path.exists(DEFAULT_FILE):
    df = load_data(DEFAULT_FILE)
else:
    st.error("Data file not found. Please upload 'SP_total_return.xlsx'.")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Global Parameters")
min_y, max_y = int(df['Year'].min()), int(df['Year'].max())

# Pool for Simulations
pool_start, pool_end = st.sidebar.slider("Historical Pool (for MC Sampling)", min_y, max_y, (min_y, max_y))
df_pool = df[(df['Year'] >= pool_start) & (df['Year'] <= pool_end)].copy()

st.sidebar.header("Inflation Settings")
inf_rate_pct = st.sidebar.slider("Annual Inflation Rate (%)", -1.0, 20.0, 2.0, step=0.1)
inf_rate = inf_rate_pct / 100.0

st.sidebar.header("Simulation Settings")
horizon = st.sidebar.slider("Horizon (Years)", 1, 50, 25)
n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)

# --- 4. DASHBOARD UI ---
st.title("S&P 500 Historical Analysis & Simulation")

# ---------------------------------------------------------
# COMPONENT 1: WHAT-IF SCENARIO
# ---------------------------------------------------------
st.divider()
st.header("1. Historical 'What-If' Scenario")
whatif_start = st.selectbox("Select Investment Starting Year", options=sorted(df['Year'].unique(), reverse=True), index=20)

# Calculate Growth
wi_data = df[df['Year'] >= whatif_start].copy()
wi_returns = wi_data['ret_decimal'].values
n_years = len(wi_returns)

# Paths (Start at 100)
wi_path = [100]
cur = 100
for r in wi_returns:
    cur *= (1 + r)
    wi_path.append(cur)

inf_path = [100 * (1 + inf_rate)**t for t in range(n_years + 1)]
x_labels = [whatif_start - 1] + list(wi_data['Year'])

col1, col2 = st.columns([2, 1])
with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x_labels, y=wi_path, name="Historical Index", line=dict(color='blue', width=3)))
    fig1.add_trace(go.Scatter(x=x_labels, y=inf_path, name=f"Inflation ({inf_rate_pct}%)", line=dict(color='red', dash='dash')))
    fig1.update_layout(yaxis_type="log", xaxis_title="Year", yaxis_title="Value (Log Scale)", height=450)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Scenario Metrics")
    final_v = wi_path[-1]
    cagr_nom = (final_v / 100)**(1/n_years) - 1 if n_years > 0 else 0
    cagr_real = (1 + cagr_nom) / (1 + inf_rate) - 1
    
    metrics = {
        "Start Value": "100.00",
        "End Value": f"{final_v:,.2f}",
        "Number of Years": str(n_years),
        "Annualized Return (Nominal)": f"{cagr_nom:.2%}",
        "Annualized Return (Real)": f"{cagr_real:.2%}"
    }
    st.table(pd.Series(metrics, name="Value"))

# ---------------------------------------------------------
# COMPONENT 2: MONTE CARLO I (BLOCK BOOTSTRAP)
# ---------------------------------------------------------
st.divider()
st.header("2. Monte Carlo I: Sequential History")
st.write(f"Randomly picks a starting year and follows the exact historical sequence for {horizon} years.")

pool_rets = df_pool['ret_decimal'].values
valid_starts = len(pool_rets) - horizon

if valid_starts < 0:
    st.warning("Historical pool is too small for this horizon.")
else:
    mc1_results = []
    for _ in range(n_sims):
        s_idx = np.random.randint(0, valid_starts + 1)
        draw = pool_rets[s_idx : s_idx + horizon]
        mc1_results.append(np.insert(100 * np.exp(np.log1p(draw).cumsum()), 0, 100))
    
    mc1_results = np.array(mc1_results)
    
    col3, col4 = st.columns([2, 1])
    with col3:
        fig2 = go.Figure()
        for i in range(min(150, n_sims)):
            fig2.add_trace(go.Scatter(y=mc1_results[i], line=dict(color='rgba(0,0,255,0.03)'), showlegend=False))
        fig2.add_trace(go.Scatter(y=np.mean(mc1_results, axis=0), name="Mean Path", line=dict(color='blue', width=4)))
        fig2.update_layout(yaxis_type="log", xaxis_title="Years", yaxis_title="Index Value (Log Scale)", height=450)
        st.plotly_chart(fig2, use_container_width=True)
        
    with col4:
        st.subheader("Simulation Stats")
        terminals = mc1_results[:, -1]
        cagrs = (terminals / 100)**(1/horizon) - 1
        stats = {
            "Mean Final Value": f"{np.mean(terminals):,.2f}",
            "Mean CAGR": f"{np.mean(cagrs):.2%}",
            "5th Percentile (Bottom)": f"{np.percentile(terminals, 5):,.2f}",
            "95th Percentile (Top)": f"{np.percentile(terminals, 95):,.2f}",
            "Prob. of Nominal Loss": f"{(terminals < 100).mean():.2%}"
        }
        st.table(pd.Series(stats, name="Value"))

# ---------------------------------------------------------
# COMPONENT 3: MONTE CARLO II (IID BOOTSTRAP)
# ---------------------------------------------------------
st.divider()
st.header("3. Monte Carlo II: Randomized Returns")
st.write(f"Randomly draws {horizon} individual years from the historical pool.")

mc2_results = []
for _ in range(n_sims):
    draw = np.random.choice(pool_rets, size=horizon, replace=True)
    mc2_results.append(np.insert(100 * np.exp(np.log1p(draw).cumsum()), 0, 100))

mc2_results = np.array(mc2_results)

col5, col6 = st.columns([2, 1])
with col5:
    fig3 = go.Figure()
    for i in range(min(150, n_sims)):
        fig3.add_trace(go.Scatter(y=mc2_results[i], line=dict(color='rgba(0,128,0,0.03)'), showlegend=False))
    fig3.add_trace(go.Scatter(y=np.mean(mc2_results, axis=0), name="Mean Path", line=dict(color='green', width=4)))
    fig3.update_layout(yaxis_type="log", xaxis_title="Years", yaxis_title="Index Value (Log Scale)", height=450)
    st.plotly_chart(fig3, use_container_width=True)

with col6:
    st.subheader("Simulation Stats")
    terminals2 = mc2_results[:, -1]
    cagrs2 = (terminals2 / 100)**(1/horizon) - 1
    stats2 = {
        "Mean Final Value": f"{np.mean(terminals2):,.2f}",
        "Mean CAGR": f"{np.mean(cagrs2):.2%}",
        "5th Percentile (Bottom)": f"{np.percentile(terminals2, 5):,.2f}",
        "95th Percentile (Top)": f"{np.percentile(terminals2, 95):,.2f}",
        "Prob. of Nominal Loss": f"{(terminals2 < 100).mean():.2%}"
    }
    st.table(pd.Series(stats2, name="Value"))