import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Annual Market MC Simulator")

# --- 2. DATA LOADING & CLEANING ---
@st.cache_data
def process_data(file_source):
    # Load data (handles local path or uploaded file)
    df = pd.read_excel(file_source, header=1)
    
    # Normalize column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Explicit mapping for Schiller_data.xlsx
    df = df.rename(columns={
        "total_return": "total_return_nominal",
        "value_real": "total_return_real_index", 
        "value": "total_return_nom_index"        
    })

    # Basic cleaning
    df = df.dropna(subset=["date"])
    df["year_int"] = df["date"].astype(int)
    
    # --- NEW: Filter for December only for the simulation ---
    # In Schiller data, December is usually represented as Year.12
    # We use np.isclose to avoid floating point precision issues
    df_annual = df[np.isclose(df['date'] % 1, 0.12)].copy()
    
    # Calculate annual returns (Dec to Dec)
    df_annual["ret_nom"] = df_annual["total_return_nom_index"].pct_change()
    df_annual["ret_real"] = df_annual["total_return_real_index"].pct_change()
    
    return df, df_annual.dropna(subset=["ret_nom", "ret_real"])

# --- 3. FILE SELECTION ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload custom Excel file", type=["xlsx"])

DEFAULT_FILE = "Schiller_data.xlsx"
data_to_load = None

if uploaded_file is not None:
    data_to_load = uploaded_file
    st.sidebar.success("Using custom uploaded data.")
elif os.path.exists(DEFAULT_FILE):
    data_to_load = DEFAULT_FILE
    st.sidebar.info("Using permanent Schiller data.")
else:
    st.error(f"Default file '{DEFAULT_FILE}' not found. Please upload a file.")
    st.stop()

# df = full monthly data (for historical plot)
# df_annual = filtered december data (for simulation)
df, df_annual = process_data(data_to_load)

# --- 4. SIDEBAR PARAMETERS ---
st.sidebar.header("Simulation Parameters")
min_y, max_y = int(df["year_int"].min()), int(df["year_int"].max())

start_year = st.sidebar.slider("Historical Plot Start Year", min_y, max_y, 1950)
end_year = st.sidebar.slider("Historical Plot End Year", start_year, max_y, max_y)

n_years = st.sidebar.slider("Investment Horizon (Years)", 1, 40, 25)
n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)

# --- 5. HELPER FUNCTIONS ---
def run_path(returns):
    # Direct compounding of returns
    return 100 * np.exp(np.log1p(returns).cumsum())

def summarize(paths, n_years):
    terminal = paths[:, -1]
    # Annualized return calculation
    ann_returns = (terminal / 100)**(1/n_years) - 1
    
    return {
        "Mean Final Value": f"{np.mean(terminal):.2f}",
        "Mean Ann. Return": f"{np.mean(ann_returns):.2%}",
        "5th Perc. Value": f"{np.percentile(terminal, 5):.2f}",
        "5th Perc. Ann. Return": f"{np.percentile(ann_returns, 5):.2%}",
        "95th Perc. Value": f"{np.percentile(terminal, 95):.2f}",
        "95th Perc. Ann. Return": f"{np.percentile(ann_returns, 95):.2%}",
        "Prob. of Nominal Loss": f"{(terminal < 100).mean():.2%}"
    }

# --- 6. MAIN PAGE ---
st.title("Annual S&P 500 Monte Carlo Analysis")
st.markdown("Simulation uses **December-to-December** annual returns.")

# (i) Historical Index Plot (Monthly)
st.header("Historical Performance (Monthly View)")
hist_data = df[(df["year_int"] >= start_year) & (df["year_int"] <= end_year)]

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=hist_data["date"], y=hist_data["total_return_nom_index"], name="Nominal Index"))
fig_hist.add_trace(go.Scatter(x=hist_data["date"], y=hist_data["total_return_real_index"], name="Real Index"))
fig_hist.update_layout(
    xaxis_title="Year", 
    yaxis_title="Index Level (Log Scale)", 
    yaxis_type="log", 
    hovermode="x unified"
)
st.plotly_chart(fig_hist, use_container_width=True)

# (ii) Monte Carlo I: Block Bootstrap (Annual)
st.header("Monte Carlo I: Annual Block Bootstrap")
st.write(f"Randomly selecting a continuous {n_years}-year historical block.")

valid_indices = df_annual.index[:-n_years]
nom_paths, real_paths = [], []

for _ in range(n_sims):
    idx = np.random.choice(valid_indices)
    window = df_annual.loc[idx : ].head(n_years) # Get the next n_years rows
    nom_paths.append(run_path(window["ret_nom"].values))
    real_paths.append(run_path(window["ret_real"].values))

nom_paths, real_paths = np.array(nom_paths), np.array(real_paths)

fig_mc1 = go.Figure()
for i in range(min(150, n_sims)):
    fig_mc1.add_trace(go.Scatter(y=nom_paths[i], line=dict(color="rgba(0,0,255,0.03)"), showlegend=False))
fig_mc1.add_trace(go.Scatter(y=nom_paths.mean(axis=0), name="Avg Nominal", line=dict(color="blue", width=3)))
fig_mc1.add_trace(go.Scatter(y=real_paths.mean(axis=0), name="Avg Real", line=dict(color="black", width=3)))
fig_mc1.update_layout(yaxis_type="log", xaxis_title="Years Elapsed", yaxis_title="Index Value (Log Scale)")
st.plotly_chart(fig_mc1, use_container_width=True)

st.table(pd.DataFrame({
    "Nominal Results": summarize(nom_paths, n_years), 
    "Real Results": summarize(real_paths, n_years)
}))

# (iii) Monte Carlo II: IID Annual Bootstrap
st.header("Monte Carlo II: Independent (IID) Annual Bootstrap")
st.write(f"Randomly drawing {n_years} independent annual returns.")

ann_ret_nom = df_annual["ret_nom"].values
ann_ret_real = df_annual["ret_real"].values

p2_nom, p2_real = [], []
for _ in range(n_sims):
    draw_n = np.random.choice(ann_ret_nom, n_years)
    draw_r = np.random.choice(ann_ret_real, n_years)
    p2_nom.append(run_path(draw_n))
    p2_real.append(run_path(draw_r))

p2_nom, p2_real = np.array(p2_nom), np.array(p2_real)

fig_mc2 = go.Figure()
for i in range(min(150, n_sims)):
    fig_mc2.add_trace(go.Scatter(y=p2_nom[i], line=dict(color="rgba(0,128,0,0.03)"), showlegend=False))
fig_mc2.add_trace(go.Scatter(y=p2_nom.mean(axis=0), name="Avg Nominal", line=dict(color="green", width=3)))
fig_mc2.add_trace(go.Scatter(y=p2_real.mean(axis=0), name="Avg Real", line=dict(color="black", width=3)))
fig_mc2.update_layout(yaxis_type="log", xaxis_title="Years Elapsed", yaxis_title="Index Value (Log Scale)")
st.plotly_chart(fig_mc2, use_container_width=True)

st.table(pd.DataFrame({
    "Nominal Results": summarize(p2_nom, n_years), 
    "Real Results": summarize(p2_real, n_years)
}))