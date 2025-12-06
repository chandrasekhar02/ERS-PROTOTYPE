import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

# Optional libs (graceful fallback if not installed)
try:
    import plotly.express as px
except Exception:
    px = None

try:
    from streamlit_lottie import st_lottie
    import requests
except Exception:
    st_lottie = None
    requests = None

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_OK = True
except Exception:
    AGGRID_OK = False

st.set_page_config(page_title="ERS Prototype", layout="wide", initial_sidebar_state="expanded")

# CSS 
st.markdown(
    """
    <style>
    .title {font-size:28px; font-weight:700; color:#0b4b6f; margin-bottom:0px;}
    .sub {color:#555; margin-top:0px; margin-bottom:12px;}
    .card {background:#fff; border-radius:8px; padding:12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);}
    .badge-high {background:#e63946;color:white;padding:6px 10px;border-radius:8px;font-weight:600;}
    .badge-medium {background:#f4d03f;color:#222;padding:6px 10px;border-radius:8px;font-weight:600;}
    .badge-low {background:#2b7cff;color:white;padding:6px 10px;border-radius:8px;font-weight:600;}
    .small {font-size:12px; color:#6b6b6b;}
    </style>
    """,
    unsafe_allow_html=True
)

# Header 
colh1, colh2 = st.columns([1,4])
with colh1:
    if st_lottie and requests:
        try:
            lottie_url = "https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json"
            r = requests.get(lottie_url, timeout=5)
            if r.status_code == 200:
                st_lottie(r.json(), height=120, key="lottie_ers")
        except Exception:
            pass
with colh2:
    st.markdown('<div class="title">Early Risk Signals (ERS) — Prototype</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Early Risk Signals for enhanced customer risk assessment.</div>', unsafe_allow_html=True) 

st.markdown("---")
st.markdown("**Load data** — upload a CSV in the sidebar or click *Use sample dataset*. The app tolerates a range of column names.")


# Column normalization & helpers

COLUMN_MAP = {
    'Customer ID': 'customer_id', 'CustomerID': 'customer_id', 'customer id': 'customer_id',
    'Credit Limit': 'credit_limit', 'creditlimit': 'credit_limit',
    'Utilisation %': 'util_pct_raw', 'Utilization %': 'util_pct_raw',
    'Utilisation': 'util_pct_raw', 'Util %': 'util_pct_raw',
    'Avg Payment Ratio': 'avg_payment_ratio_raw', 'Average Payment Ratio': 'avg_payment_ratio_raw',
    'Avg Payment %': 'avg_payment_ratio_raw',
    'Min Due Paid Frequency': 'min_due_freq_raw', 'Min Due Frequency': 'min_due_freq_raw',
    'Merchant Mix Index': 'merchant_mix_index', 'Merchant Mix': 'merchant_mix_index',
    'Cash Withdrawal %': 'cash_withdrawal_pct_raw', 'Cash Withdrawal': 'cash_withdrawal_pct_raw',
    'Recent Spend Change %': 'recent_spend_change_pct_raw', 'Recent Spend Change': 'recent_spend_change_pct_raw',
    'DPD Bucket Next Month': 'dpd_bucket_next_month', 'dpdbucketnextmonth': 'dpd_bucket_next_month'
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        clean = col.strip()
        if clean in COLUMN_MAP:
            rename_map[col] = COLUMN_MAP[clean]
        else:
            for k, v in COLUMN_MAP.items():
                if clean.lower() == k.lower():
                    rename_map[col] = v
    return df.rename(columns=rename_map)

def to_decimal_percent(series):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = series.copy().astype(str).str.replace('%','').str.strip()
    s = pd.to_numeric(s, errors='coerce').fillna(0)
    return s.apply(lambda x: x/100 if x > 1 else x)


@st.cache_data(ttl=60*30)
def compute_flags(df_serializable: pd.DataFrame) -> pd.DataFrame:
    """
    Compute flags and ERS score with a single DPD parameter:
      - P7_dpd_severity: DPD=0 -> 0, DPD=1 -> 2, DPD>=2 -> 4
    Final ERS_score = weighted sum of flags (includes P7 as numeric points).
    Tier: 0-3 Low, 4-5 Medium, 6+ High
    """
    df = df_serializable.copy()

    def get_series(col, default=0):
        if col in df:
            s = df[col]
            return s if isinstance(s, pd.Series) else pd.Series(s, index=df.index)
        return pd.Series([default] * len(df), index=df.index)

    def to_decimal_percent_local(series):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        s = series.astype(str).str.replace('%', '').str.strip()
        s = pd.to_numeric(s, errors='coerce').fillna(0)
        return s.apply(lambda x: x/100 if x > 1 else x)

    # core metrics
    df['util_pct'] = to_decimal_percent_local(get_series('util_pct_raw'))
    df['avg_payment_ratio'] = to_decimal_percent_local(get_series('avg_payment_ratio_raw'))
    df['min_due_freq'] = to_decimal_percent_local(get_series('min_due_freq_raw'))
    df['cash_withdrawal_pct'] = to_decimal_percent_local(get_series('cash_withdrawal_pct_raw'))
    df['recent_spend_change_pct'] = to_decimal_percent_local(get_series('recent_spend_change_pct_raw'))
    df['merchant_mix_index'] = pd.to_numeric(get_series('merchant_mix_index'), errors='coerce').fillna(0)

    # Original flags P1..P6
    df['P1_utilisation'] = (df['util_pct'] >= 0.90).astype(int)
    df['P2_low_commit'] = (df['avg_payment_ratio'] <= 0.40).astype(int)
    df['P3_min_payment_trap'] = (df['min_due_freq'] >= 0.75).astype(int)
    df['P4_liquidity_stress'] = (df['cash_withdrawal_pct'] >= 0.15).astype(int)
    df['P5_sudden_shock'] = (df['recent_spend_change_pct'] <= -0.20).astype(int)
    df['P6_concentrated_spending'] = (df['merchant_mix_index'] <= 0.30).astype(int)

    # ---------------- P7 = DPD Severity (ONLY ONE DPD RULE) ----------------
    # Use 'dpd_bucket_next_month' (expected in dataset) to create P7
    if 'dpd_bucket_next_month' in df.columns:
        dpd_numeric = pd.to_numeric(df['dpd_bucket_next_month'], errors='coerce').fillna(0).astype(int)
    else:
        dpd_numeric = pd.Series([0]*len(df), index=df.index)

    # P7 stores numeric points (0,2,4)
    df['P7_dpd_severity'] = 0
    df.loc[dpd_numeric == 1, 'P7_dpd_severity'] = 2
    df.loc[dpd_numeric >= 2, 'P7_dpd_severity'] = 4

    # ---------------- P8 overlimit behavior (kept) ----------------
    overlimit_cols = ['overlimit_flag', 'is_overlimit', 'over_limit']
    overlimit_series = pd.Series([0] * len(df), index=df.index)
    found_flag = False
    for c in overlimit_cols:
        if c in df.columns:
            overlimit_series = df[c].astype(str).str.lower().isin(['1','true','yes','y','t']).astype(int)
            found_flag = True
            break
    if not found_flag:
        overlimit_series = (df['util_pct'] > 1.0).astype(int)
    df['P8_overlimit_behavior'] = overlimit_series.astype(int)

    # weights (tuned)
    w = {
        'P1_utilisation': 3,
        'P2_low_commit': 2,
        'P3_min_payment_trap': 2,
        'P4_liquidity_stress': 1,
        'P5_sudden_shock': 2,
        'P6_concentrated_spending': 1,
        'P7_dpd_severity': 1,   # P7 already contains numeric points, so weight=1
        'P8_overlimit_behavior': 2
    }

    # compute ERS score explicitly using the components
    df['ERS_score'] = (
        df['P1_utilisation'] * w['P1_utilisation'] +
        df['P2_low_commit'] * w['P2_low_commit'] +
        df['P3_min_payment_trap'] * w['P3_min_payment_trap'] +
        df['P4_liquidity_stress'] * w['P4_liquidity_stress'] +
        df['P5_sudden_shock'] * w['P5_sudden_shock'] +
        df['P6_concentrated_spending'] * w['P6_concentrated_spending'] +
        df['P7_dpd_severity'] * w['P7_dpd_severity'] +    # adds 0/2/4
        df['P8_overlimit_behavior'] * w['P8_overlimit_behavior']
    )

    # final tier mapping: 0-3 Low, 4-5 Medium, 6+ High
    def tier_map(s):
        if s >= 6:
            return 'High'
        if s >= 4:
            return 'Medium'
        return 'Low'

    df['ERS_risk_tier'] = df['ERS_score'].apply(tier_map)

    # store dpd numeric for display
    df['dpd_bucket_next_month'] = dpd_numeric

    # label if present
    df['delinq_label'] = (dpd_numeric > 0).astype(int)

    return df

# session-state safe loader

if 'df' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['loaded_from'] = None

# Sidebar controls
with st.sidebar:
    st.header("Data & Controls")
    uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'], key="uploader")
    if st.button("Use sample dataset"):
        sample = Path(__file__).parent / "sample_ers_input.csv"
        if sample.exists():
            st.session_state['df'] = pd.read_csv(sample)
            st.session_state['loaded_from'] = 'sample'
        else:
            st.error("Sample file not found.")
    if uploaded is not None:
        try:
            st.session_state['df'] = pd.read_csv(uploaded)
            st.session_state['loaded_from'] = 'uploaded'
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

    if st.button("Clear data"):
        st.session_state['df'] = None
        st.session_state['loaded_from'] = None
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("**Customer quick search**")
    quick_customer = st.text_input("Customer ID (sidebar)", value="", key="quick_search")
    st.markdown("---")
    st.markdown("**Export Options**")
    topk = st.number_input("Export top K risky", min_value=1, max_value=1000, value=20, step=1)
    st.markdown("")

# Use session df
df = st.session_state.get('df', None)
if df is None:
    st.info("Upload a CSV or click 'Use sample dataset' from the sidebar to begin.")
    st.stop()

# Normalize & persist
df = normalize_columns(df)
st.session_state['df'] = df

# Compute flags & tiers (cached)
with st.spinner("Computing ERS flags and applying DPD logic..."):
    df_flags = compute_flags(df)

# Colour maps
COLOR_MAP = {'High': '#e63946', 'Medium': '#f4d03f', 'Low': '#2b7cff'}
PLOTLY_COLOR_MAP = {'High': '#e63946', 'Medium': '#f4d03f', 'Low': '#2b7cff'}

# Tabs
tab_overview, tab_customer, tab_logs = st.tabs(["Overview", "Customer", "Logs"])

#overview Tab
with tab_overview:
    st.header("Overview")
    colA, colB, colC, colD = st.columns([1.2,1,1,1])
    colA.metric("Total customers", len(df_flags))
    colB.metric("High risk", int((df_flags['ERS_risk_tier']=='High').sum()))
    colC.metric("Medium risk", int((df_flags['ERS_risk_tier']=='Medium').sum()))
    colD.metric("Low risk", int((df_flags['ERS_risk_tier']=='Low').sum()))

    st.markdown("**Risk tier distribution**")
    if px:
        fig = px.pie(df_flags, names='ERS_risk_tier', title='Risk Tier Distribution', hole=0.4,
                     color_discrete_map=PLOTLY_COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(df_flags['ERS_risk_tier'].value_counts())

    st.markdown("**ERS Score vs Utilisation**")
    if px:
        fig2 = px.scatter(df_flags, x='util_pct', y='ERS_score', color='ERS_risk_tier',
                          hover_data=['customer_id','credit_limit','avg_payment_ratio','dpd_bucket_next_month'],
                          labels={'util_pct':'Utilisation (decimal)','ERS_score':'ERS score'},
                          color_discrete_map=PLOTLY_COLOR_MAP)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("Install plotly for interactive charts: `pip install plotly`")

    st.markdown("**Top risky customers (by ERS score)**")
    topk_df = df_flags.sort_values('ERS_score', ascending=False).head(int(topk))
    st.write(f"Showing top {len(topk_df)}")
    csv_bytes = topk_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download top-K CSV", data=csv_bytes, file_name="topk_ers.csv", mime="text/csv")

    st.markdown("**Portfolio table**")
    display_cols = ['customer_id','credit_limit','util_pct','avg_payment_ratio','ERS_score','ERS_risk_tier','dpd_bucket_next_month','P7_dpd_severity']
    if AGGRID_OK:
        gb = GridOptionsBuilder.from_dataframe(df_flags[display_cols])
        gb.configure_default_column(filter=True, sortable=True, resizable=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gridOptions = gb.build()
        AgGrid(df_flags[display_cols], gridOptions=gridOptions, theme='alpine', height=320)
    else:
        st.dataframe(df_flags[display_cols])

# Customer Tab
with tab_customer:
    st.header("Customer Detail")
    cust_input = quick_customer if quick_customer else st.text_input("Enter customer_id (e.g., C001)", value="", key="cust_input")
    if cust_input:
        row = df_flags[df_flags['customer_id'].astype(str) == cust_input]
        if row.empty:
            st.warning("Customer not found. Check ID or upload matching dataset.")
        else:
            r = row.iloc[0]
            col1, col2 = st.columns([2,2])
            with col1:
                st.subheader(f"Customer: {r['customer_id']}")
                st.metric("ERS score", r['ERS_score'])
                tier = r['ERS_risk_tier']
                if tier == 'High':
                    st.markdown(f"<span class='badge-high'>High</span>", unsafe_allow_html=True)
                elif tier == 'Medium':
                    st.markdown(f"<span class='badge-medium'>Medium</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='badge-low'>Low</span>", unsafe_allow_html=True)

                st.write("**Key recent metrics**")
                kdf = pd.DataFrame({
                    "metric":["util_pct","avg_payment_ratio","min_due_freq","cash_withdrawal_pct","recent_spend_change_pct","merchant_mix_index","dpd_bucket_next_month","P7_dpd_severity"],
                    "value":[r.get('util_pct',np.nan), r.get('avg_payment_ratio',np.nan), r.get('min_due_freq',np.nan),
                             r.get('cash_withdrawal_pct',np.nan), r.get('recent_spend_change_pct',np.nan), r.get('merchant_mix_index',np.nan),
                             r.get('dpd_bucket_next_month', np.nan), r.get('P7_dpd_severity', np.nan)]
                })
                st.table(kdf.set_index('metric'))
            with col2:
                st.write("### Active Flags")
                # show any P* column with non-zero value (P7 may be 2 or 4)
                flags = [c for c in row.columns if str(c).startswith('P') and float(r.get(c,0)) != 0]
                if flags:
                    for f in flags:
                        st.write(f"- {f}: {r.get(f)}")
                else:
                    st.write("No active flags.")
                st.write("### Recommended action")
                if r['ERS_risk_tier']=='High':
                    st.info("High Risk — Priority outreach: phone call by RM / collections; discuss EMI/hardship options.")
                elif r['ERS_risk_tier']=='Medium':
                    st.warning("Medium Risk — Soft outreach: SMS reminder + in-app nudge; consider EMI conversion.")
                else:
                    st.success("Low Risk — No immediate action; provide educational nudges.")

                # Outreach logging
                st.write("### Outreach simulation")
                action = st.selectbox("Action", ["No action","Send SMS (log)","Call & Log Note"])
                note = st.text_area("Note")

                if st.button("Execute outreach action"):
                    log_path = Path(__file__).parent / "outreach_log.csv"
                    entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "customer_id": r['customer_id'],
                        "ERS_score": r['ERS_score'],
                        "ERS_tier": r['ERS_risk_tier'],
                        "action": action,
                        "note": note
                    }
                    if log_path.exists():
                        prev = pd.read_csv(log_path)
                        prev = pd.concat([prev, pd.DataFrame([entry])], ignore_index=True)
                        prev.to_csv(log_path, index=False)
                    else:
                        pd.DataFrame([entry]).to_csv(log_path, index=False)
                    st.success("Action logged.")

# Logs Tab
with tab_logs:
    st.header("Outreach Logs")
    log_path = Path(__file__).parent / "outreach_log.csv"
    if log_path.exists():
        logs = pd.read_csv(log_path)
        logs = logs.sort_values('timestamp', ascending=False)
        st.write("Recent outreach actions (latest 10)")
        st.table(logs.head(10))
        st.download_button("Download All Logs (CSV)", data=logs.to_csv(index=False).encode('utf-8'),
                           file_name="outreach_log.csv", mime="text/csv")
    else:
        st.info("No outreach logs yet. Execute an action from the Customer tab to create logs.")

st.markdown("---")

