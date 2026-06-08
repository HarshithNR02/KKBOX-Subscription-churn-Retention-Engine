# app/utils.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import streamlit as st
import os
import requests
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

DATA_DIR = Path.home() / "kkbox-churn" / "data" / "parquet"
MODEL_DIR = Path.home() / "kkbox-churn" / "models"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "lgbm_churn_final.pkl")

@st.cache_resource
def load_isotonic():
    return joblib.load(MODEL_DIR / "isotonic_calibrator.pkl")

@st.cache_data
def load_master():
    master = pd.read_parquet(DATA_DIR / "master_dataset.parquet")
    return master

@st.cache_data
def load_segments():
    return pd.read_parquet(DATA_DIR / "user_segments.parquet")

@st.cache_data
def load_shap_values():
    return pd.read_parquet(DATA_DIR / "shap_values.parquet")

def get_risk_tier(prob):
    if prob < 0.05:
        return "🟢 Low Risk", "green"
    elif prob < 0.20:
        return "🟡 Medium Risk", "orange"
    elif prob < 0.50:
        return "🟠 High Risk", "darkorange"
    else:
        return "🔴 Critical Risk", "red"

def get_retention_action(segment, risk_tier):
    actions = {
        'Lost':           "❌ Write off — CAC exceeds CLV. No action recommended.",
        'Mid_Value':      "💰 Send targeted discount — 1 month free or price lock offer.",
        'High_Engage':    "⬆️ Premium upsell — offer hi-fi audio or offline downloads.",
        'Returning':      "🎵 Engagement campaign — personalized playlist recommendations.",
        'Short_Cycle':    "✅ No action needed — auto-renewing, highest CLV segment.",
    }
    return actions.get(segment, "📊 Monitor closely.")

FEATURE_COLS = [
    'current_auto_renew', 'current_is_cancel', 'current_plan_days',
    'current_payment_method', 'days_since_last_transaction',
    'days_since_last_cancel', 'customer_tenure_days', 'days_to_expiry',
    'total_transactions', 'total_cancellations', 'auto_renew_rate',
    'cancel_rate', 'total_amount_paid', 'avg_amount_paid', 'pay_per_day',
    'avg_plan_days', 'max_plan_days', 'has_long_plan', 'payment_methods_used',
    'total_discount', 'discount_flag', 'cancelled_last_30_days',
    'cancel_last3', 'autorenew_last3', 'cancel_last5', 'autorenew_last5',
    'price_change', 'has_price_increase', 'payment_method_changed',
    'auto_renew_switch', 'switched_off_flag', 'avg_interval_days',
    'nopay_ratio', 't1_amount', 't2_amount', 'last_churn', 'last_not_churn',
    'last_not_in', 'auto_not_cancel_ratio', 'plan_changes',
    'city', 'registered_via', 'bd_clean', 'age_known', 'gender_known',
    'unknown_gender_young', 'account_age_days', 'is_city_1',
    'is_invalid_registration', 'registration_year', 'bd_bucket',
    'days_since_last_listen', 'active_days_15d', 'active_days_30d',
    'active_days_60d', 'active_days_120d', 'active_days_all',
    'secs_15d', 'secs_30d', 'secs_all', 'songs_15d', 'songs_30d',
    'songs_all', 'completion_rate_all', 'completion_rate_15d',
    'unq_per_day_all', 'song_per_day_all', 'listening_trend_ratio',
    'active_days_trend', 'stopped_listening', 'listener_tenure_days',
    'has_listening_15d', 'has_listening_30d', 'final_day_songs',
    'final_day_secs', 'songs_std_30d', 'song_per_day_30d',
    'song_per_day_15d', 'num_unq_15d', 'num_unq_30d', 'num_100_15d',
    'num_100_30d', 'num_25_15d', 'num_25_30d', 'prev_active_days_15d',
    'prev_secs_15d', 'prev_songs_15d', 'final_day_completion_rate',
    'final_day_skip_rate', 'songs_std_15d', 'secs_std_30d', 'secs_std_15d',
    'skip_rate_all', 'has_listening_history', 'days_between_last_listen_and_expiry',
    'listening_per_dollar', 'is_extreme_risk', 'is_silent_loyal',
    'auto_off_inactive', 'listen_to_plan_ratio', 'active_days_per_sub_day'
]

def get_ai_insights(churn_prob, segment, risk_drivers, clv, user_features):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "⚠️ Add OPENAI_API_KEY to .env to enable AI insights."

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        prompt = f"""You are a senior customer retention analyst at a music streaming service like Spotify.

A customer has been flagged by our churn prediction model:
- Churn Probability: {churn_prob:.1%}
- Customer Segment: {segment}
- Estimated CLV: {clv:,.0f} TWD
- Risk Drivers: {', '.join(risk_drivers) if risk_drivers else 'None identified'}
- Days Since Last Transaction: {user_features['days_since_last_transaction']}
- Days Since Last Listen: {user_features['days_since_last_listen']}
- Auto Renew: {'ON' if user_features['current_auto_renew'] == 1 else 'OFF'}
- Tenure: {user_features['customer_tenure_days']} days
- Churned Last Month: {'Yes' if user_features['last_churn'] == 1 else 'No'}

Write a concise 3-paragraph retention analysis:
1. Why this customer is at risk based on behavioral signals
2. Specific retention action with business justification
3. ROI of intervention vs cost of losing this customer

Be specific, data-driven, actionable. Under 200 words."""

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI insights unavailable: {str(e)}"
