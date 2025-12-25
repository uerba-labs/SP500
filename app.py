import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(layout="wide",     page_title="S&P 500 Monte Carlo Simulator | Understanding Investment Risk",
    page_icon="📈")

# --- 2. DATA LOADING ---
@st.cache_data
def load_data(file_source):
    df = pd.read_excel(file_source, header=1)
    df = df[['Year', 'Total return']].dropna()
    df['Year'] = df['Year'].astype(int)
    df = df.sort_values('Year').reset_index(drop=True)
    df['ret_decimal'] = df['Total return'] / 100.0
    return df

DEFAULT_FILE = "SP_total_return.xlsx"
if os.path.exists(DEFAULT_FILE):
    df = load_data(DEFAULT_FILE)
else:
    st.error("Please ensure 'SP_total_return.xlsx' is in your GitHub folder.")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Global Parameters")
min_y, max_y = int(df['Year'].min()), int(df['Year'].max())

st.sidebar.header("Inflation Settings")
inf_rate_pct = st.sidebar.slider("Expected Annual Inflation (%)", -1.0, 20.0, 2.0, step=0.1)
inf_rate = inf_rate_pct / 100.0

st.sidebar.header("Simulation Settings")
horizon = st.sidebar.slider("Horizon (Years)", 1, 50, 25)
n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)

# --- 4. HELPER FUNCTIONS ---
def get_stats_table(paths, horizon, inf_rate):
    terminals = paths[:, -1]
    cagrs = (terminals / 100)**(1/horizon) - 1
    
    # Calculate inflation benchmark
    inf_benchmark = 100 * (1 + inf_rate)**horizon
    
    # Probabilities as "X out of 100"
    nom_loss_freq = round((terminals < 100).mean() * 100)
    real_loss_freq = round((terminals < inf_benchmark).mean() * 100)
    
    data = {
        "Metric": ["Final Index Value", "Annualized Return (CAGR)"],
        "Mean": [f"{np.mean(terminals):,.2f}", f"{np.mean(cagrs):.2%}"],
        "5th Percentile (Bottom)": [f"{np.percentile(terminals, 5):,.2f}", f"{np.percentile(cagrs, 5):.2%}"],
        "95th Percentile (Top)": [f"{np.percentile(terminals, 95):,.2f}", f"{np.percentile(cagrs, 95):.2%}"]
    }
    
    return pd.DataFrame(data), nom_loss_freq, real_loss_freq

# --- 5. DASHBOARD UI ---
st.title("How uncertain are long-term stock market returns? A Monte Carlo simulator based on historical S&P 500 total returns")

# Text Placeholder 1
intro_text = st.text_area("Introduction", 
             "This interactive tool illustrates how long-term investment outcomes can vary, even when based on the same historical data. Using S&P 500 index historical total return data (including dividends), the simulator generates thousands of possible future paths to help build intuition about risk, compounding, inflation, and uncertainty. This is an educational tool — not a forecast.")

# Text Placeholder 2
intro_text = st.text_area("How to read these simulations", 
             "Each line represents a possible future path for a $100 investment. All simulations start at the same value, but evolve differently depending on the sequence of returns. The shaded areas show the range of typical outcomes, while the median line represents the middle scenario. Large differences between paths highlight why long-term outcomes are uncertain, even when average returns appear attractive. This interactive tool illustrates how long-term investment outcomes can vary, even when based on the same historical data. ")

# ---------------------------------------------------------
# COMPONENT 1: HISTORICAL WHAT-IF
# ---------------------------------------------------------
st.divider()
st.header("1. The Power of History: What-If?")
st_col1, st_col2 = st.columns([2, 1])

with st_col1:
    whatif_start = st.selectbox("If you had invested €100 in...", options=sorted(df['Year'].unique(), reverse=True), index=20)
    wi_data = df[df['Year'] >= whatif_start].copy()
    wi_returns = wi_data['ret_decimal'].values
    n_yrs = len(wi_returns)
    
    wi_path = [100]
    cur = 100
    for r in wi_returns:
        cur *= (1 + r)
        wi_path.append(cur)
    
    inf_path = [100 * (1 + inf_rate)**t for t in range(n_yrs + 1)]
    x_labels = [whatif_start - 1] + list(wi_data['Year'])

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x_labels, y=wi_path, name="Historical Index", line=dict(color='blue', width=3)))
    fig1.add_trace(go.Scatter(x=x_labels, y=inf_path, name="Inflation Baseline", line=dict(color='red', dash='dash')))
    fig1.update_layout(yaxis_type="log", xaxis_title="Year", yaxis_title="Value (Log Scale)", height=400)
    st.plotly_chart(fig1, use_container_width=True)

with st_col2:
    st.subheader("Historical Results")
    final_v = wi_path[-1]
    cagr_nom = (final_v / 100)**(1/n_yrs) - 1 if n_yrs > 0 else 0
    cagr_real = (1 + cagr_nom) / (1 + inf_rate) - 1
    
    st.write(f"**Years Elapsed:** {n_yrs}")
    st.write(f"**Final Value:** €{final_v:,.2f}")
    st.write(f"**Annualized Return (Nominal):** {cagr_nom:.2%}")
    st.write(f"**Annualized Return (Real):** {cagr_real:.2%}")
    st.info("Note: This is the average annual rate of return resulting from this investment. The annualized return accounts for the rate of inflation considered in the analysis.")

# ---------------------------------------------------------
# COMPONENT 2: MC I (SEQUENTIAL)
# ---------------------------------------------------------
st.divider()
st.header("2. Monte Carlo I: Historical Sequences")
mc1_desc = st.text_input("Description", "Monte Carlo I preserves history. Each simulation starts in a randomly chosen historical year and then follows the actual sequence of returns observed over the selected horizon. This approach keeps real-world features such as market crashes, prolonged downturns,and recovery periods.")
pool_rets = df['ret_decimal'].values
valid_starts = len(pool_rets) - horizon

if valid_starts >= 0:
    mc1_results = []
    for _ in range(n_sims):
        s_idx = np.random.randint(0, valid_starts + 1)
        draw = pool_rets[s_idx : s_idx + horizon]
        mc1_results.append(np.insert(100 * np.exp(np.log1p(draw).cumsum()), 0, 100))
    mc1_results = np.array(mc1_results)
    
    col3, col4 = st.columns([2, 1])
    with col3:
        fig2 = go.Figure()
        for i in range(min(100, n_sims)):
            fig2.add_trace(go.Scatter(y=mc1_results[i], line=dict(color='rgba(0,0,255,0.03)'), showlegend=False))
        # Add Mean Path
        fig2.add_trace(go.Scatter(y=np.mean(mc1_results, axis=0), name="Mean Path", line=dict(color='blue', width=4)))
        # Add Inflation Line
        mc_inf_line = [100 * (1 + inf_rate)**t for t in range(horizon + 1)]
        fig2.add_trace(go.Scatter(y=mc_inf_line, name="Inflation", line=dict(color='red', dash='dash')))
        
        fig2.update_layout(yaxis_type="log", xaxis_title="Years", yaxis_title="Index (Log Scale)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        df_stats, nom_p, real_p = get_stats_table(mc1_results, horizon, inf_rate)
        st.table(df_stats)
        st.metric("Prob. of Nominal Loss", f"{nom_p:.1f} out of 100")
        st.metric("Prob. of Real Loss (Below Inflation)", f"{real_p:.1f} out of 100")

# ---------------------------------------------------------
# COMPONENT 3: MC II (RANDOMIZED)
# ---------------------------------------------------------
st.divider()
st.header("3. Monte Carlo II: Pure Randomness")
mc2_desc = st.text_input("MC II Description", "Here, annual returns are drawn independently from history and applied sequentially. This assumes that each year is independent of the previous one and typically produces a wider range of outcomes.")

mc2_results = []
for _ in range(n_sims):
    draw = np.random.choice(pool_rets, size=horizon, replace=True)
    mc2_results.append(np.insert(100 * np.exp(np.log1p(draw).cumsum()), 0, 100))
mc2_results = np.array(mc2_results)

col5, col6 = st.columns([2, 1])
with col5:
    fig3 = go.Figure()
    for i in range(min(100, n_sims)):
        fig3.add_trace(go.Scatter(y=mc2_results[i], line=dict(color='rgba(0,128,0,0.03)'), showlegend=False))
    fig3.add_trace(go.Scatter(y=np.mean(mc2_results, axis=0), name="Mean Path", line=dict(color='green', width=4)))
    fig3.add_trace(go.Scatter(y=mc_inf_line, name="Inflation", line=dict(color='red', dash='dash')))
    fig3.update_layout(yaxis_type="log", xaxis_title="Years", yaxis_title="Index (Log Scale)", height=400)
    st.plotly_chart(fig3, use_container_width=True)

with col6:
    df_stats2, nom_p2, real_p2 = get_stats_table(mc2_results, horizon, inf_rate)
    st.table(df_stats2)
    st.metric("Prob. of Nominal Loss", f"{nom_p2:.1f} out of 100")
    st.metric("Prob. of Real Loss (Below Inflation)", f"{real_p2:.1f} out of 100")

# --- 6. FOOTER / DISCLAIMERS ---
st.divider() # Adds a horizontal line to separate the tool from the footer

st.markdown("""
### **Disclaimers**
This tool is provided for **educational purposes only**. It does not constitute financial advice, investment recommendations, or an offer to buy or sell any financial instrument.

**Past performance does not guarantee future results.**

The simulations are based on historical S&P 500 data, including dividends and inflation adjustments derived from publicly available sources. Data accuracy is not guaranteed, and results depend on modelling assumptions.

The author assumes no responsibility for decisions made based on this tool. Users remain fully responsible for their own investment decisions.

---

### **Privacy**
If you choose to subscribe, your email address will be processed by a third-party email provider solely for the purpose of receiving updates about this project. You can unsubscribe at any time.

No personal data is sold or shared for marketing purposes.
""")