# app/pages/2_Customer_Segments.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_master

st.set_page_config(page_title="Customer Segments", page_icon="👥", layout="wide")
st.title("👥 Customer Segmentation")
st.markdown("RFM-based K-Means clustering — 5 behavioral segments identified from 970K users.")
st.divider()

master = load_master()

# Segment colors
COLORS = {
    'Lost': '#e74c3c',
    'Mid_Value': '#f39c12',
    'High_Engage': '#2ecc71',
    'Returning': '#3498db',
    'Short_Cycle': '#9b59b6'
}

# --- Segment Overview ---
st.subheader("📊 Segment Overview")

seg_stats = master.groupby('segment').agg(
    Users=('msno', 'count'),
    Churn_Rate=('is_churn', 'mean'),
    Avg_CLV=('clv', 'mean'),
    Median_CLV=('clv', 'median'),
    Avg_Tenure=('customer_tenure_days', 'mean'),
    Avg_Monthly_Revenue=('monthly_revenue', 'mean'),
    Revenue_At_Risk=('clv', lambda x: (x * master.loc[x.index, 'churn_prob']).mean())
).round(2).sort_values('Churn_Rate', ascending=False)

seg_stats['Churn_Rate'] = seg_stats['Churn_Rate'].apply(lambda x: f"{x:.1%}")
seg_stats['Avg_CLV'] = seg_stats['Avg_CLV'].apply(lambda x: f"{x:,.0f} TWD")
seg_stats['Median_CLV'] = seg_stats['Median_CLV'].apply(lambda x: f"{x:,.0f} TWD")
seg_stats['Avg_Tenure'] = seg_stats['Avg_Tenure'].apply(lambda x: f"{x:.0f} days")
seg_stats['Avg_Monthly_Revenue'] = seg_stats['Avg_Monthly_Revenue'].apply(lambda x: f"{x:.0f} TWD")

st.dataframe(seg_stats, use_container_width=True)

st.divider()

# --- Charts Row ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Users per Segment")
    counts = master['segment'].value_counts()
    colors = [COLORS.get(s, '#95a5a6') for s in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.85)
    ax.set_ylabel("Number of Users")
    ax.set_title("User Distribution by Segment")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("📉 Churn Rate by Segment")
    churn_rates = master.groupby('segment')['is_churn'].mean().sort_values(ascending=False)
    colors = [COLORS.get(s, '#95a5a6') for s in churn_rates.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(churn_rates.index, churn_rates.values * 100, color=colors, alpha=0.85)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Segment")
    ax.axhline(y=master['is_churn'].mean() * 100, color='black',
               linestyle='--', alpha=0.5, label=f"Overall: {master['is_churn'].mean():.1%}")
    ax.legend()
    for bar, val in zip(bars, churn_rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# --- CLV by Segment ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("💰 Average CLV by Segment")
    clv_by_seg = master.groupby('segment')['clv'].mean().sort_values(ascending=False)
    colors = [COLORS.get(s, '#95a5a6') for s in clv_by_seg.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(clv_by_seg.index, clv_by_seg.values, color=colors, alpha=0.85)
    ax.set_ylabel("Average CLV (TWD)")
    ax.set_title("Customer Lifetime Value by Segment")
    for bar, val in zip(bars, clv_by_seg.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

with col4:
    st.subheader("⚠️ Revenue at Risk by Segment")
    master['expected_loss'] = master['clv'] * master['churn_prob']
    risk_by_seg = master.groupby('segment')['expected_loss'].sum().sort_values(ascending=False)
    colors = [COLORS.get(s, '#95a5a6') for s in risk_by_seg.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(risk_by_seg.index, risk_by_seg.values / 1e6, color=colors, alpha=0.85)
    ax.set_ylabel("Revenue at Risk (Million TWD)")
    ax.set_title("Expected Revenue Loss by Segment")
    for bar, val in zip(bars, risk_by_seg.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val/1e6:.0f}M', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# --- Segment Deep Dive ---
st.subheader("🔍 Segment Deep Dive")
selected_seg = st.selectbox("Select Segment", master['segment'].unique())

seg_data = master[master['segment'] == selected_seg]

d1, d2, d3, d4, d5 = st.columns(5)
with d1:
    st.metric("Users", f"{len(seg_data):,}")
with d2:
    st.metric("Churn Rate", f"{seg_data['is_churn'].mean():.1%}")
with d3:
    st.metric("Avg CLV", f"{seg_data['clv'].mean():,.0f} TWD")
with d4:
    st.metric("Avg Tenure", f"{seg_data['customer_tenure_days'].mean():.0f} days")
with d5:
    st.metric("Avg Monthly Rev", f"{seg_data['monthly_revenue'].mean():.0f} TWD")

# Segment description
descriptions = {
    'Lost': "🔴 **Lost** — 82% churn probability. Long-promo plan users who paid upfront and disappeared. No retention action recommended — CAC exceeds CLV.",
    'Mid_Value': "🟡 **Mid_Value** — Moderate engagement, 6.2% actual churn. Largest revenue segment. Targeted discount offers recommended.",
    'High_Engage': "🟢 **High_Engage** — Heavy listeners (5x average listening time). Low churn risk. Premium upsell opportunity.",
    'Returning': "🔵 **Returning** — Re-engaged users with listening spike. Nurture with personalized recommendations.",
    'Short_Cycle': "🟣 **Short_Cycle** — Expiring soon but auto-renewing. Highest CLV, lowest risk. No intervention needed."
}
st.info(descriptions.get(selected_seg, ""))