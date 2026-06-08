# app/pages/4_Retention_Strategy.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_master, get_ai_insights

st.set_page_config(page_title="Retention Strategy", page_icon="💡", layout="wide")
st.title("💡 Retention Strategy")
st.markdown("Data-driven retention recommendations — who to target, what to offer, and expected ROI.")
st.divider()

master = load_master()
master['expected_loss'] = master['clv'] * master['churn_prob']

# --- Revenue at Risk Summary ---
st.subheader("💰 Revenue at Risk Summary")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Users", f"{len(master):,}")
with m2:
    total_risk = master['expected_loss'].sum()
    st.metric("Total Revenue at Risk", f"{total_risk/1e9:.2f}B TWD")
with m3:
    high_risk = master[master['churn_prob'] > 0.5]
    st.metric("Critical Risk Users", f"{len(high_risk):,}")
with m4:
    actionable = master[(master['churn_prob'] > 0.05) & (master['clv'] > 10000)]
    st.metric("Actionable Users", f"{len(actionable):,}")

st.divider()

# --- Retention Priority Matrix ---
st.subheader("🎯 Retention Priority Matrix")
st.markdown("**X-axis:** Churn Probability | **Y-axis:** Customer Lifetime Value | **Size:** Number of users")

# Sample for plotting
sample = master.sample(min(5000, len(master)), random_state=42)

COLORS = {
    'Lost': '#e74c3c',
    'Mid_Value': '#f39c12',
    'High_Engage': '#2ecc71',
    'Returning': '#3498db',
    'Short_Cycle': '#9b59b6'
}

fig, ax = plt.subplots(figsize=(10, 6))
for seg in sample['segment'].unique():
    seg_data = sample[sample['segment'] == seg]
    ax.scatter(
        seg_data['churn_prob'],
        seg_data['clv'].clip(upper=200000),
        c=COLORS.get(seg, '#95a5a6'),
        alpha=0.4,
        s=10,
        label=seg
    )

# Quadrant lines
ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=50000, color='gray', linestyle='--', alpha=0.5)

# Quadrant labels
ax.text(0.02, 180000, "🟣 PROTECT\n(High Value, Low Risk)", fontsize=9, color='purple', alpha=0.8)
ax.text(0.25, 180000, "🔴 SAVE NOW\n(High Value, High Risk)", fontsize=9, color='red', alpha=0.8)
ax.text(0.02, 5000, "✅ MONITOR\n(Low Value, Low Risk)", fontsize=9, color='green', alpha=0.8)
ax.text(0.25, 5000, "❌ WRITE OFF\n(Low Value, High Risk)", fontsize=9, color='gray', alpha=0.8)

ax.set_xlabel("Churn Probability", fontsize=11)
ax.set_ylabel("Customer Lifetime Value (TWD)", fontsize=11)
ax.set_title("Retention Priority Matrix", fontsize=13)
ax.legend(loc='center right', markerscale=3)
plt.tight_layout()
st.pyplot(fig)

st.divider()

# --- Strategy per Segment ---
st.subheader("📋 Retention Strategy by Segment")

strategies = {
    'Lost': {
        'icon': '🔴',
        'users': len(master[master['segment'] == 'Lost']),
        'churn': f"{master[master['segment'] == 'Lost']['is_churn'].mean():.1%}",
        'clv': f"{master[master['segment'] == 'Lost']['clv'].mean():,.0f} TWD",
        'action': 'Write Off',
        'offer': 'None — CAC exceeds CLV',
        'budget': '0 TWD per user',
        'reason': 'Average CLV is 727 TWD. Any retention offer would cost more than the customer is worth.'
    },
    'Mid_Value': {
        'icon': '🟡',
        'users': len(master[master['segment'] == 'Mid_Value']),
        'churn': f"{master[master['segment'] == 'Mid_Value']['is_churn'].mean():.1%}",
        'clv': f"{master[master['segment'] == 'Mid_Value']['clv'].mean():,.0f} TWD",
        'action': 'Targeted Discount',
        'offer': '1 month free or 20% price lock for 3 months',
        'budget': '~150 TWD per user',
        'reason': 'CLV 52K TWD. Cost of offer (150 TWD) << expected revenue saved (52K × 6.2% = 3,224 TWD).'
    },
    'High_Engage': {
        'icon': '🟢',
        'users': len(master[master['segment'] == 'High_Engage']),
        'churn': f"{master[master['segment'] == 'High_Engage']['is_churn'].mean():.1%}",
        'clv': f"{master[master['segment'] == 'High_Engage']['clv'].mean():,.0f} TWD",
        'action': 'Premium Upsell',
        'offer': 'Hi-fi audio quality or offline downloads feature',
        'budget': '~50 TWD per user',
        'reason': 'Heavy listeners (506K secs/month). Low churn risk but high engagement = upsell opportunity.'
    },
    'Returning': {
        'icon': '🔵',
        'users': len(master[master['segment'] == 'Returning']),
        'churn': f"{master[master['segment'] == 'Returning']['is_churn'].mean():.1%}",
        'clv': f"{master[master['segment'] == 'Returning']['clv'].mean():,.0f} TWD",
        'action': 'Engagement Campaign',
        'offer': 'Personalized playlist + social features access',
        'budget': '~30 TWD per user',
        'reason': 'Re-engaged users with listening spike (trend ratio 234x). Nurture before they drift again.'
    },
    'Short_Cycle': {
        'icon': '🟣',
        'users': len(master[master['segment'] == 'Short_Cycle']),
        'churn': f"{master[master['segment'] == 'Short_Cycle']['is_churn'].mean():.1%}",
        'clv': f"{master[master['segment'] == 'Short_Cycle']['clv'].mean():,.0f} TWD",
        'action': 'No Action',
        'offer': 'None — auto-renewing reliably',
        'budget': '0 TWD per user',
        'reason': 'Highest CLV (68K TWD), lowest churn risk (0.05%). Intervention wastes budget.'
    }
}

for seg, info in strategies.items():
    with st.expander(f"{info['icon']} {seg} — {info['users']:,} users | Churn: {info['churn']} | Avg CLV: {info['clv']}"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Action:** {info['action']}")
            st.markdown(f"**Offer:** {info['offer']}")
        with c2:
            st.markdown(f"**Budget:** {info['budget']}")
        with c3:
            st.markdown(f"**Reasoning:** {info['reason']}")

st.divider()

# --- ROI Calculator ---
st.subheader("📊 Retention ROI Calculator")
st.markdown("Estimate the return on investment for a retention campaign.")

col_a, col_b = st.columns(2)
with col_a:
    target_segment = st.selectbox("Target Segment", list(strategies.keys()))
    offer_cost = st.number_input("Cost per Retention Offer (TWD)", 0, 1000, 150)
    expected_save_rate = st.slider("Expected Save Rate (%)", 1, 50, 15)

seg_data = master[master['segment'] == target_segment]
n_users = len(seg_data)
churn_prob_mean = seg_data['churn_prob'].mean()
clv_mean = seg_data['clv'].mean()

users_to_target = int(n_users * churn_prob_mean)
campaign_cost = users_to_target * offer_cost
users_saved = int(users_to_target * expected_save_rate / 100)
revenue_saved = users_saved * clv_mean
roi = ((revenue_saved - campaign_cost) / campaign_cost * 100) if campaign_cost > 0 else 0

with col_b:
    st.metric("Users to Target", f"{users_to_target:,}")
    st.metric("Campaign Cost", f"{campaign_cost:,.0f} TWD")
    st.metric("Expected Users Saved", f"{users_saved:,}")
    st.metric("Expected Revenue Saved", f"{revenue_saved:,.0f} TWD")
    st.metric("ROI", f"{roi:.0f}%", delta=f"{'Profitable' if roi > 0 else 'Loss'}")

st.divider()

# --- AI Strategy Advisor ---
st.subheader("🤖 AI Strategy Advisor")
st.markdown("Get GPT-4o powered retention strategy recommendations.")

if st.button("Generate AI Strategy", type="primary"):
    with st.spinner("Generating strategy..."):
        summary = f"""
Segment: {target_segment}
Users at risk: {users_to_target:,}
Average CLV: {clv_mean:,.0f} TWD
Churn probability: {churn_prob_mean:.1%}
Campaign cost: {campaign_cost:,.0f} TWD
Expected ROI: {roi:.0f}%
"""
        prompt_context = {
            'days_since_last_transaction': int(seg_data['days_since_last_transaction'].mean()),
            'days_since_last_listen': int(seg_data['days_since_last_listen'].mean()) if seg_data['days_since_last_listen'].notna().any() else 30,
            'current_auto_renew': int(seg_data['current_auto_renew'].mean().round()),
            'customer_tenure_days': int(seg_data['customer_tenure_days'].mean()),
            'last_churn': int(seg_data['last_churn'].mean().round())
        }
        ai_response = get_ai_insights(
            churn_prob_mean,
            target_segment,
            [f"Campaign cost: {campaign_cost:,.0f} TWD", f"Expected ROI: {roi:.0f}%"],
            clv_mean,
            prompt_context
        )
    st.write(ai_response)