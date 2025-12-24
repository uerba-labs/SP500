import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Market MC Simulator")

# --- 2. DATA LOADING & CLEANING ---
@st.cache_data
def process_data(file_source):
    # Handles both local file paths and Streamlit UploadedFile objects
    df = pd.read_excel(file_source, header=1)
    
    # Normalize column names for robust matching
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Explicit mapping for the specific Schiller_data.xlsx structure provided
    df = df.rename(columns={
        "total_return": "total_return_nominal",
        "value_real": "total_return_real_index", # Maps 'Value_Real'
        "value": "total_return_nom_index"        # Maps 'Value'
    })

    # Basic cleaning and date handling
    df = df.dropna(subset=["date"])
    df["year_int"] = df["date"].astype(int)
    
    # Pre-calculate monthly returns to optimize the simulation loop
    df["ret_nom"] = df["total_return_nom_index"].pct_change()
    df["ret_real"] = df["total_return_real_index"].pct_change()
    
    return df

# --- 3. FILE SELECTION LOGIC ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload custom Excel file", type=["xlsx"])

DEFAULT_FILE = "Schiller_data.xlsx"
data_to_load = None

# Prioritize uploaded file; fall back to permanent Schiller file
if uploaded_file is not None:
    data_to_load = uploaded_file
    st.sidebar.success("Using custom uploaded data.")
elif os.path.exists(DEFAULT_FILE):
    data_to_load = DEFAULT_FILE
    st.sidebar.info("Using permanent Schiller data.")
else:
    st.error(f"Default file '{DEFAULT_FILE}' not found. Please upload a file.")
    st.stop()

# Execute loading
df = process_data(data_to_load)

# --- 4. SIDEBAR SIMULATION PARAMETERS ---
st.sidebar.header("Simulation Parameters")
min_y, max_y = int(df["year_int"].min()), int(df["year_int"].max())

start_year = st.sidebar.slider("Historical Plot Start Year", min_y, max_y, 1950)
end_year = st.sidebar.slider("Historical Plot End Year", start_year, max_y, max_y)

n_years = st.sidebar.slider("Investment Horizon (Years)", 1, 40, 20)
n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)

# --- 5. MAIN PAGE LAYOUT ---
st.title("S&P 500 Historical & Monte Carlo Analysis")
st.markdown("""
This tool analyzes S&P 500 performance using historical data. 
**Monte Carlo I** reshuffles historical monthly blocks, while **Monte Carlo II** draws random annual returns.
""")

# (i) Historical Index Plot
st.header("Historical Performance")
hist_data = df[(df["year_int"] >= start_year) & (df["year_int"] <= end_year)]

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=hist_data["date"], y=hist_data["total_return_nom_index"], name="Nominal Index"))
fig_hist.add_trace(go.Scatter(x=hist_data["date"], y=hist_data["total_return_real_index"], name="Real (Inflation Adj) Index"))
fig_hist.update_layout(xaxis_title="Year", yaxis_title="Index Level", hovermode="x unified")
st.plotly_chart(fig_hist, use_container_width=True)

# --- 6. SIMULATION ENGINES ---
def run_path(returns):
    # Vectorized compounding for speed
    return 100 * np.exp(np.log1p(returns).cumsum())

def summarize(paths):
    terminal = paths[:, -1]
    return {
        "Mean Final Value": f"{np.mean(terminal):.2f}",
        "5th Percentile": f"{np.percentile(terminal, 5):.2f}",
        "95th Percentile": f"{np.percentile(terminal, 95):.2f}",
        "Prob. of Nominal Loss": f"{(terminal < 100).mean():.2%}"
    }

# (ii) MC I: Block Bootstrap
st.header("Monte Carlo I: Block Bootstrap (Path Dependent)")
valid_indices = df.index[:- (n_years * 12)]
nom_paths, real_paths = [], []

for _ in range(n_sims):
    start_idx = np.random.choice(valid_indices)
    window = df.iloc[start_idx : start_idx + (n_years * 12)]
    nom_paths.append(run_path(window["ret_nom"].values))
    real_paths.append(run_path(window["ret_real"].values))

nom_paths, real_paths = np.array(nom_paths), np.array(real_paths)

# Plot MC I
fig_mc1 = go.Figure()
for i in range(min(150, n_sims)): # Plot subset for performance
    fig_mc1.add_trace(go.Scatter(y=nom_paths[i], line=dict(color="rgba(0,0,255,0.03)"), showlegend=False))
fig_mc1.add_trace(go.Scatter(y=nom_paths.mean(axis=0), name="Avg Nominal", line=dict(color="blue", width=3)))
fig_mc1.add_trace(go.Scatter(y=real_paths.mean(axis=0), name="Avg Real", line=dict(color="black", width=3)))
st.plotly_chart(fig_mc1, use_container_width=True)
st.table(pd.DataFrame({"Nominal Results": summarize(nom_paths), "Real Results": summarize(real_paths)}))

# (iii) MC II: Annual Return Bootstrap
st.header("Monte Carlo II: IID Annual Return Bootstrap")
annual_df = df.groupby("year_int").last()
ann_ret_nom = annual_df["total_return_nom_index"].pct_change().dropna().values
ann_ret_real = annual_df["total_return_real_index"].pct_change().dropna().values

p2_nom, p2_real = [], []
for _ in range(n_sims):
    draw_n = np.random.choice(ann_ret_nom, n_years)
    draw_r = np.random.choice(ann_ret_real, n_years)
    
    # Map annual returns back to a monthly-step visual path
    m_nom = np.repeat((1 + draw_n)**(1/12) - 1, 12)
    m_real = np.repeat((1 + draw_r)**(1/12) - 1, 12)
    
    p2_nom.append(run_path(m_nom))
    p2_real.append(run_path(m_real))

p2_nom, p2_real = np.array(p2_nom), np.array(p2_real)

# Plot MC II
fig_mc2 = go.Figure()
for i in range(min(150, n_sims)):
    fig_mc2.add_trace(go.Scatter(y=p2_nom[i], line=dict(color="rgba(0,128,0,0.03)"), showlegend=False))
fig_mc2.add_trace(go.Scatter(y=p2_nom.mean(axis=0), name="Avg Nominal", line=dict(color="green", width=3)))
fig_mc2.add_trace(go.Scatter(y=p2_real.mean(axis=0), name="Avg Real", line=dict(color="black", width=3)))
st.plotly_chart(fig_mc2, use_container_width=True)
st.table(pd.DataFrame({"Nominal Results": summarize(p2_nom), "Real Results": summarize(p2_real)}))