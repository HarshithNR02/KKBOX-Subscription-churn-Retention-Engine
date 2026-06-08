# app/pages/3_WhatIf_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_model, load_isotonic, get_risk_tier, get_retention_action, get_ai_insights, FEATURE_COLS

st.set_page_config(page_title="What-If Analysis", page_icon="🧪", layout="wide")
st.title("🧪 What-If Analysis")
st.markdown("Manually adjust user features to explore how behavioral changes affect churn risk.")
st.divider()

model = load_model()
iso_reg = load_isotonic()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Transaction Features")
    days_since_last_transaction = st.slider("Days Since Last Transaction", 0, 200, 14)
    days_to_expiry = st.slider("Days to Expiry", -30, 200, 15)
    current_auto_renew = st.selectbox("Current Auto Renew", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    current_is_cancel = st.selectbox("Current Is Cancel", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    total_transactions = st.number_input("Total Transactions", 1, 100, 12)
    total_cancellations = st.number_input("Total Cancellations", 0, 20, 0)
    customer_tenure_days = st.number_input("Customer Tenure (days)", 0, 3000, 500)
    avg_amount_paid = st.number_input("Avg Amount Paid (TWD)", 0, 2000, 149)
    auto_renew_rate = st.slider("Auto Renew Rate", 0.0, 1.0, 0.9)
    cancel_rate = st.slider("Cancel Rate", 0.0, 1.0, 0.02)
    last_churn = st.selectbox("Churned Last Month", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col_right:
    st.subheader("🎵 Listening Features")
    days_since_last_listen = st.slider("Days Since Last Listen", 0, 200, 2)
    active_days_30d = st.slider("Active Days (Last 30d)", 0, 31, 18)
    active_days_15d = st.slider("Active Days (Last 15d)", 0, 15, 10)
    secs_30d = st.number_input("Listening Seconds (Last 30d)", 0, 500000, 100000)
    listening_trend_ratio = st.number_input("Listening Trend Ratio (Mar/Feb)", 0.0, 10.0, 1.0)
    completion_rate_15d = st.slider("Completion Rate (15d)", 0.0, 1.0, 0.72)
    stopped_listening = st.selectbox("Stopped Listening in March", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    has_listening_history = st.selectbox("Has Listening History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

if st.button("Predict Churn", type="primary", use_container_width=True):
    input_data = {col: 0 for col in FEATURE_COLS}
    input_data.update({
        'days_since_last_transaction': days_since_last_transaction,
        'days_to_expiry': days_to_expiry,
        'current_auto_renew': current_auto_renew,
        'current_is_cancel': current_is_cancel,
        'total_transactions': total_transactions,
        'total_cancellations': total_cancellations,
        'customer_tenure_days': customer_tenure_days,
        'avg_amount_paid': avg_amount_paid,
        'auto_renew_rate': auto_renew_rate,
        'cancel_rate': cancel_rate,
        'last_churn': last_churn,
        'days_since_last_listen': days_since_last_listen,
        'active_days_30d': active_days_30d,
        'active_days_15d': active_days_15d,
        'secs_30d': secs_30d,
        'listening_trend_ratio': listening_trend_ratio,
        'completion_rate_15d': completion_rate_15d,
        'stopped_listening': stopped_listening,
        'has_listening_history': has_listening_history,
        'auto_renew_switch': 'always_on' if current_auto_renew == 1 and auto_renew_rate == 1.0 else 'off',
        'city': 1,
        'registered_via': 7,
        'bd_bucket': '25_34',
        't1_amount': avg_amount_paid,
        't2_amount': avg_amount_paid,
        'avg_plan_days': 30,
        'current_plan_days': 30,
        'pay_per_day': avg_amount_paid / 30,
        'total_amount_paid': avg_amount_paid * total_transactions,
        'auto_not_cancel_ratio': auto_renew_rate * (1 - cancel_rate),
        'last_not_churn': 1 - last_churn,
        'last_not_in': 0,
    })

    input_df = pd.DataFrame([input_data])
    for col in ['city', 'registered_via', 'bd_bucket', 'auto_renew_switch']:
        input_df[col] = input_df[col].astype('category')

    raw_prob = model.predict_proba(input_df[FEATURE_COLS])[:, 1][0]
    cal_prob = raw_prob
    risk_label, _ = get_risk_tier(cal_prob)

    if cal_prob > 0.5:
        segment = 'Lost'
    elif active_days_30d >= 25:
        segment = 'High_Engage'
    elif customer_tenure_days > 600 and auto_renew_rate > 0.8:
        segment = 'Short_Cycle'
    elif listening_trend_ratio > 5:
        segment = 'Returning'
    else:
        segment = 'Mid_Value'

    drivers = []
    if days_since_last_transaction > 30:
        drivers.append(f"No transaction in {days_since_last_transaction} days")
    if current_auto_renew == 0:
        drivers.append("Auto-renew is OFF")
    if last_churn == 1:
        drivers.append("Churned last month")
    if stopped_listening == 1:
        drivers.append("Stopped listening in March")
    if days_to_expiry < 7:
        drivers.append(f"Membership expires in {days_to_expiry} days")

    # Store in session state
    st.session_state['whatif_result'] = {
        'cal_prob': cal_prob,
        'risk_label': risk_label,
        'segment': segment,
        'clv': min(avg_amount_paid / max(cal_prob, 0.001), 201588),
        'action': get_retention_action(segment, risk_label),
        'drivers': drivers,
        'user_features': {
            'days_since_last_transaction': days_since_last_transaction,
            'days_since_last_listen': days_since_last_listen,
            'current_auto_renew': current_auto_renew,
            'customer_tenure_days': customer_tenure_days,
            'last_churn': last_churn
        }
    }

# Display results from session state — persists across button clicks
if 'whatif_result' in st.session_state:
    res = st.session_state['whatif_result']

    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Churn Probability", f"{res['cal_prob']:.1%}")
    with r2:
        st.metric("Risk Tier", res['risk_label'])
    with r3:
        st.metric("Estimated Segment", res['segment'])
    with r4:
        st.metric("Estimated CLV", f"{res['clv']:,.0f} TWD")

    st.divider()
    st.subheader("💡 Recommended Action")
    st.info(res['action'])

    st.subheader("🔑 Key Risk Drivers")
    if res['drivers']:
        for d in res['drivers']:
            st.write(f"{d}")
    else:
        st.write("✅ No major risk signals detected")

    st.divider()
    st.subheader("AI Retention Analyst")
    if st.button("Generate AI Insights", type="secondary"):
        with st.spinner("Generating AI insights..."):
            ai_response = get_ai_insights(
                res['cal_prob'], res['segment'],
                res['drivers'], res['clv'],
                res['user_features']
            )
        st.session_state['whatif_ai'] = ai_response

    if 'whatif_ai' in st.session_state:
        st.write(st.session_state['whatif_ai'])