# app/pages/3_SHAP_Explainability.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_master, load_shap_values, load_model

st.set_page_config(page_title="SHAP Explainability", layout="wide")
st.title("Model Explainability")
st.markdown("SHAP (SHapley Additive exPlanations) — understanding why the model predicts churn.")
st.divider()

MODEL_DIR = Path.home() / "kkbox-churn" / "models"

# --- Global Feature Importance ---
st.subheader("🌍 Global Feature Importance")
st.markdown("Mean absolute SHAP values across 5,000 validation users — higher = more influential.")

shap_df = load_shap_values()
mean_shap = shap_df.abs().mean().sort_values(ascending=False).head(20)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e74c3c' if v > 1.0 else '#f39c12' if v > 0.3 else '#3498db'
              for v in mean_shap.values]
    ax.barh(mean_shap.index[::-1], mean_shap.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top 20 Features by SHAP Importance")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("**Top 5 Insights:**")
    st.markdown("""
    🔴 **days_since_last_transaction** (3.85)  
    Most dominant signal — users who haven't transacted in 30+ days are near-certain churners.
    
    🟠 **days_to_expiry** (1.02)  
    Expiring soon = high risk. Plans with 0-7 days remaining show 97% churn.
    
    🟡 **avg_amount_paid** (0.57)  
    Higher payers are more loyal — price sensitivity signal.
    
    🟡 **auto_not_cancel_ratio** (0.36)  
    Users who always auto-renew and never cancel = lowest churn risk.
    
    🟢 **customer_tenure_days** (0.31)  
    Longer tenure = lower churn. 790-day users have 0.06% churn probability.
    """)

st.divider()

# --- Waterfall Plots ---
st.subheader("Individual Prediction Explanations")
st.markdown("Waterfall plots show exactly why the model predicted churn for specific users.")

tab1, tab2 = st.tabs(["🔴 High Risk User", "🟢 Low Risk User"])

with tab1:
    st.markdown("**User Profile:** 99.5% churn probability — actual churner ✅")
    st.markdown("""
    - `days_since_last_transaction = 36` → **+5.29 SHAP** (36 days no transaction)
    - `autorenew_last3 = 0` → **+1.01 SHAP** (no auto-renew in last 3 transactions)  
    - `days_to_expiry = 55` → **+0.83 SHAP** (plan expiring soon)
    - `auto_not_cancel_ratio = 0` → **-1.20 SHAP** (reduces prediction slightly)
    - **Final prediction: f(x) = 5.275 → 99.5% churn probability**
    """)
    if (MODEL_DIR / "shap_waterfall_highrisk.png").exists():
        st.image(str(MODEL_DIR / "shap_waterfall_highrisk.png"))

with tab2:
    st.markdown("**User Profile:** 0.06% churn probability — actual non-churner ✅")
    st.markdown("""
    - `days_since_last_transaction = 0` → **-4.45 SHAP** (transacted today)
    - `customer_tenure_days = 790` → **-0.40 SHAP** (long-term subscriber)
    - `auto_renew_rate = 1.0` → **-0.14 SHAP** (always auto-renews)
    - `auto_not_cancel_ratio = 1.0` → **-0.20 SHAP** (never cancels)
    - **Final prediction: f(x) = -7.472 → 0.06% churn probability**
    """)
    if (MODEL_DIR / "shap_waterfall_lowrisk.png").exists():
        st.image(str(MODEL_DIR / "shap_waterfall_lowrisk.png"))

st.divider()

# --- SHAP Summary Plot ---
st.subheader("SHAP Summary Plot")
st.markdown("Each dot is a user. Red = high feature value, Blue = low feature value. Position shows impact on churn prediction.")
if (MODEL_DIR / "shap_summary.png").exists():
    st.image(str(MODEL_DIR / "shap_summary.png"), use_container_width=True)

st.divider()

# --- Feature Distribution by Churn ---
st.subheader("Feature Distribution Analysis")
master = load_master()

selected_feature = st.selectbox(
    "Select feature to analyze",
    ['days_since_last_transaction', 'days_to_expiry', 'avg_amount_paid',
     'customer_tenure_days', 'auto_renew_rate', 'active_days_30d',
     'listening_trend_ratio', 'secs_30d']
)

fig, ax = plt.subplots(figsize=(10, 4))
churned = master[master['is_churn'] == 1][selected_feature].dropna()
retained = master[master['is_churn'] == 0][selected_feature].dropna()

# Clip for visualization
p99 = master[selected_feature].quantile(0.99)
churned = churned.clip(upper=p99)
retained = retained.clip(upper=p99)

ax.hist(retained, bins=50, alpha=0.6, color='#2ecc71', label='Retained', density=True)
ax.hist(churned, bins=50, alpha=0.6, color='#e74c3c', label='Churned', density=True)
ax.set_xlabel(selected_feature)
ax.set_ylabel("Density")
ax.set_title(f"Distribution of {selected_feature} by Churn Status")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

col_a, col_b = st.columns(2)
with col_a:
    st.metric("Churned — Mean", f"{churned.mean():.2f}")
    st.metric("Churned — Median", f"{churned.median():.2f}")
with col_b:
    st.metric("Retained — Mean", f"{retained.mean():.2f}")
    st.metric("Retained — Median", f"{retained.median():.2f}")