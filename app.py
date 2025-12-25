import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import streamlit.components.v1 as components

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="S&P 500 Simulator | Historical Reality vs. Future Uncertainty",
    page_icon="📈"
)

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
inf_rate_pct = st.sidebar.slider("Annual Inflation Rate (%)", -1.0, 20.0, 2.0, step=0.1, help="Inflation erodes your purchasing power. This rate defines the 'Real' growth threshold.")
inf_rate = inf_rate_pct / 100.0

st.sidebar.header("Simulation Settings")
horizon = st.sidebar.slider("Investment Horizon (Years)", 1, 50, 25, help="The number of years you plan to stay invested.")
n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100, help="More simulations provide a more comprehensive view of potential 'fat tail' events.")

# --- 4. HELPER FUNCTIONS ---
def get_stats_table(paths, horizon, inf_rate):
    terminals = paths[:, -1]
    cagrs = (terminals / 100)**(1/horizon) - 1
    inf_benchmark = 100 * (1 + inf_rate)**horizon
    
    nom_loss_freq = round((terminals < 100).mean() * 100)
    real_loss_freq = round((terminals < inf_benchmark).mean() * 100)
    
    data = {
        "Scenario": ["Average Outcome", "Pessimistic (Bottom 5%)", "Optimistic (Top 5%)"],
        "Final Index Value": [f"{np.mean(terminals):,.2f}", f"{np.percentile(terminals, 5):,.2f}", f"{np.percentile(terminals, 95):,.2f}"],
        "Annualized Return (CAGR)": [f"{np.mean(cagrs):.2%}", f"{np.percentile(cagrs, 5):.2%}", f"{np.percentile(cagrs, 95):.2%}"]
    }
    
    return pd.DataFrame(data), nom_loss_freq, real_loss_freq

# --- 5. DASHBOARD UI ---
st.title("Historical Fairy Tales vs. Monte Carlo Reality")

st.markdown("""
Looking at historical stock market charts often feels like reading a fairy tale—a steady climb to wealth. 
However, history is just one path that *did* happen. To invest wisely, we must explore the thousands of 
alternate paths that *could* happen, especially the "fat tails" where risk hides.
""")

with st.expander("🎓 New to investing? Read this first"):
    st.write("""
        **Nominal vs. Real:** Nominal is the money in your account; Real is what that money can actually buy. 
        **Fat Tails:** The statistical reality that extreme market crashes happen more often than standard math predicts.
        **CAGR:** The steady annual growth rate required to reach your final balance.
    """)

# ---------------------------------------------------------
# COMPONENT 1: HISTORICAL WHAT-IF
# ---------------------------------------------------------
st.divider()
st.header("1. The Historical 'Fairy Tale'", help="See how a single lump sum would have grown in the past.")
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
    st.subheader("Hindsight Results")
    final_v = wi_path[-1]
    cagr_nom = (final_v / 100)**(1/n_yrs) - 1 if n_yrs > 0 else 0
    cagr_real = (1 + cagr_nom) / (1 + inf_rate) - 1
    
    st.write(f"**Duration:** {n_yrs} years")
    st.write(f"**Final Value:** €{final_v:,.2f}")
    st.write(f"**Nominal Return:** {cagr_nom:.2%}")
    st.write(f"**Real Return:** {cagr_real:.2%}")
    st.info("History looks easy because the 'bad paths' didn't happen to you. The simulations below show what you might have faced instead.")

# ---------------------------------------------------------
# COMPONENT 2: MC I (SEQUENTIAL)
# ---------------------------------------------------------
st.divider()
st.header("2. Monte Carlo I: Sequential Cycles", help="This method picks random blocks of history, preserving the 'order' of crashes and recoveries.")
st.info("💡 **Why this matters:** It tests if your strategy survives prolonged downturns followed by recoveries (path dependency).")

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
        fig2.add_trace(go.Scatter(y=np.mean(mc1_results, axis=0), name="Average Path", line=dict(color='blue', width=4)))
        mc_inf_line = [100 * (1 + inf_rate)**t for t in range(horizon + 1)]
        fig2.add_trace(go.Scatter(y=mc_inf_line, name="Inflation Baseline", line=dict(color='red', dash='dash')))
        fig2.update_layout(yaxis_type="log", xaxis_title="Years", yaxis_title="Index (Log Scale)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        df_stats, nom_p, real_p = get_stats_table(mc1_results, horizon, inf_rate)
        st.table(df_stats)
        st.metric("Risk of Nominal Loss", f"{nom_p} out of 100")
        st.metric("Risk of Real Loss", f"{real_p} out of 100", help="Scenarios that end below the red dashed inflation line.")

# ---------------------------------------------------------
# COMPONENT 3: MC II (RANDOMIZED)
# ---------------------------------------------------------
st.divider()
st.header("3. Monte Carlo II: Randomized Uncertainty", help="This shuffles all historical years into a random 'salad.'")
st.info("💡 **Why this matters:** This explores 'Fat Tails'—scenarios where bad years happen more frequently than they did in our single version of history.")

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
    fig3.add_trace(go.Scatter(y=np.mean(mc2_results, axis=0), name="Average Path", line=dict(color='green', width=4)))
    fig3.add_trace(go.Scatter(y=mc_inf_line, name="Inflation Baseline", line=dict(color='red', dash='dash')))
    fig3.update_layout(yaxis_type="log", xaxis_title="Years", yaxis_title="Index (Log Scale)", height=400)
    st.plotly_chart(fig3, use_container_width=True)

with col6:
    df_stats2, nom_p2, real_p2 = get_stats_table(mc2_results, horizon, inf_rate)
    st.table(df_stats2)
    st.metric("Risk of Nominal Loss", f"{nom_p2} out of 100")
    st.metric("Risk of Real Loss", f"{real_p2} out of 100")

# --- 6. FOOTER / DISCLAIMERS ---
st.divider() 
st.markdown("""
### **Disclaimers**
This tool is for **educational purposes only**. It does not constitute financial advice. **Past performance does not guarantee future results.** The simulations are based on historical S&P 500 data. Data accuracy is not guaranteed. 
Users remain fully responsible for their own investment decisions.
""")