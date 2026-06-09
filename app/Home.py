# app/Home.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_master, load_segments

st.set_page_config(
    page_title="KKBOX Churn Intelligence",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 KKBOX Churn Intelligence Platform")
st.markdown("**Predicting customer churn for Asia's leading music streaming service**")
st.divider()

# Load data
master = load_master()
segments = load_segments()

# --- KPI Row ---
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Users", f"{len(master):,}")
with col2:
    st.metric("Overall Churn Rate", f"{master['is_churn'].mean():.1%}")
with col3:
    st.metric("Model AUC", "0.9481")
with col4:
    st.metric("Features Engineered", "101")
with col5:
    st.metric("Est. Revenue at Risk", "~7.9B TWD")

st.divider()

# --- Two columns layout ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Churn Rate by Segment")
    seg_stats = master.groupby('segment').agg(
        Users=('msno', 'count'),
        Churn_Rate=('is_churn', 'mean'),
        Avg_CLV=('clv', 'mean')
    ).round(4).sort_values('Churn_Rate', ascending=False)
    seg_stats['Churn_Rate'] = seg_stats['Churn_Rate'].apply(lambda x: f"{x:.1%}")
    seg_stats['Avg_CLV'] = seg_stats['Avg_CLV'].apply(lambda x: f"{x:,.0f} TWD")
    st.dataframe(seg_stats, use_container_width=True)

with col_right:
    st.subheader("🔑 Top Churn Signals (SHAP)")
    signals = {
        'days_since_last_transaction': 3.85,
        'days_to_expiry': 1.02,
        'avg_amount_paid': 0.57,
        'current_payment_method': 0.37,
        'auto_not_cancel_ratio': 0.36,
        'customer_tenure_days': 0.31,
        'pay_per_day': 0.20,
        'autorenew_last3': 0.15,
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    features = list(signals.keys())
    values = list(signals.values())
    ax.barh(features[::-1], values[::-1], color='crimson', alpha=0.7)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Top Features by SHAP Importance')
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# --- Project Summary ---
st.subheader("🏗️ Project Architecture")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("**Data Engineering**\n\n250M row logs processed with DuckDB out-of-core. 3 tables joined. 101 features engineered.")
with col2:
    st.info("**Modeling**\n\nLightGBM with Optuna tuning (50 trials). Temporal train/val split. AUC 0.9481 — top 4-5% competition equivalent.")
with col3:
    st.info("**Segmentation**\n\nK-Means RFM clustering. 5 segments: Lost, Mid_Value, High_Engage, Returning, Short_Cycle.")
with col4:
    st.info("**Business Impact**\n\n7.9B TWD revenue at risk identified. Retention priority matrix built per segment.")