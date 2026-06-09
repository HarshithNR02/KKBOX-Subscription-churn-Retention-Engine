import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_model, load_isotonic, load_master, get_risk_tier, get_retention_action, get_ai_insights, FEATURE_COLS

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("Churn Prediction Engine")
st.markdown("Production-style scoringlook up any user by ID and get instant churn analysis.")
st.divider()

model = load_model()
iso_reg = load_isotonic()
master = load_master()

@st.cache_data
def get_feature_matrix():
    drop_cols = [
        "last_transaction_date", "last_expire_date",
        "last_listen_date", "first_listen_date",
        "cluster_k5", "segment", "churn_prob", "monthly_revenue",
        "churn_prob_clipped", "clv", "retention_priority", "is_churn"
    ]
    df = master.drop(columns=[c for c in drop_cols if c in master.columns])
    for col in ["city", "registered_via", "bd_bucket", "auto_renew_switch"]:
        df[col] = df[col].astype("category")
    return df

@st.cache_data
def get_sample_users():
    low      = master[(master["churn_prob"] < 0.05) & (master["is_churn"] == 0)][["msno","segment","churn_prob","clv"]].head(25)
    medium   = master[(master["churn_prob"] >= 0.05) & (master["churn_prob"] < 0.20) & (master["is_churn"] == 1)][["msno","segment","churn_prob","clv"]].head(25)
    high     = master[(master["churn_prob"] >= 0.20) & (master["churn_prob"] < 0.50) & (master["is_churn"] == 1)][["msno","segment","churn_prob","clv"]].head(25)
    critical = master[(master["churn_prob"] >= 0.50) & (master["is_churn"] == 1)][["msno","segment","churn_prob","clv"]].head(25)
    sample = pd.concat([low, medium, high, critical])
    sample["risk_tier"] = sample["churn_prob"].apply(
        lambda x: "Low Risk"      if x < 0.05 else
                  "Medium Risk"   if x < 0.20 else
                  "High Risk"     if x < 0.50 else
                  "Critical Risk"
    )
    sample["churn_prob_pct"] = sample["churn_prob"].apply(lambda x: f"{x:.1%}")
    sample["clv_fmt"] = sample["clv"].apply(lambda x: f"{x:,.0f} TWD")
    return sample

feature_matrix = get_feature_matrix()
sample_users   = get_sample_users()

st.subheader("Select User")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🟢 Random Low Risk", use_container_width=True):
        st.session_state["selected_msno"] = master[(master["churn_prob"] < 0.05) & (master["is_churn"] == 0)]["msno"].sample(1).values[0]
        st.rerun()
with col2:
    if st.button("🟡 Random Medium Risk", use_container_width=True):
        st.session_state["selected_msno"] = master[(master["churn_prob"] >= 0.05) & (master["churn_prob"] < 0.20) & (master["is_churn"] == 1)]["msno"].sample(1).values[0]
        st.rerun()
with col3:
    if st.button("🟠 Random High Risk", use_container_width=True):
        st.session_state["selected_msno"] = master[(master["churn_prob"] >= 0.20) & (master["churn_prob"] < 0.50) & (master["is_churn"] == 1)]["msno"].sample(1).values[0]
        st.rerun()
with col4:
    if st.button("🔴 Random Critical Risk", use_container_width=True):
        st.session_state["selected_msno"] = master[(master["churn_prob"] >= 0.50) & (master["is_churn"] == 1)]["msno"].sample(1).values[0]
        st.rerun()

st.divider()

st.markdown("**Sample Users (25 from each risk tier) — click a row to select:**")
display_cols = ["msno", "risk_tier", "churn_prob_pct", "segment", "clv_fmt"]
display_df = sample_users[display_cols].rename(columns={
    "msno":           "User ID",
    "risk_tier":      "Risk Tier",
    "churn_prob_pct": "Churn Probability",
    "segment":        "Segment",
    "clv_fmt":        "Estimated CLV"
})

selected_rows = st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row"
)

if selected_rows and selected_rows.selection.rows:
    row_idx = selected_rows.selection.rows[0]
    st.session_state["selected_msno"] = sample_users.iloc[row_idx]["msno"]

st.divider()

user_id = st.text_input(
    "Or enter any User ID directly (supports all 970K users):",
    value=st.session_state.get("selected_msno", ""),
    placeholder="paste msno here..."
)

if user_id and user_id.strip():
    user_id = user_id.strip()
    if user_id not in master["msno"].values:
        st.error("User ID not found in dataset.")
    else:
        user_row      = master[master["msno"] == user_id].iloc[0]
        user_features = feature_matrix[feature_matrix["msno"] == user_id].drop(columns=["msno"])

        # Use stored churn_prob for consistency with sample table
        stored_prob = user_row.get("churn_prob")
        churn_prob  = float(stored_prob) if pd.notna(stored_prob) else float(model.predict_proba(user_features[FEATURE_COLS])[:, 1][0])

        risk_label, _ = get_risk_tier(churn_prob)
        segment    = user_row.get("segment", "Unknown")
        clv        = user_row.get("clv", user_row.get("avg_amount_paid", 149) / max(churn_prob, 0.001))
        action     = get_retention_action(segment, risk_label)

        st.divider()
        st.subheader("Prediction Results")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        with r2:
            st.metric("Risk Tier", risk_label)
        with r3:
            st.metric("Segment", segment)
        with r4:
            st.metric("Estimated CLV", f"{clv:,.0f} TWD")

        st.divider()
        st.subheader("User Profile")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Transaction Signals**")
            st.write(f"Days since last transaction: **{user_row.get('days_since_last_transaction', 'N/A')}**")
            st.write(f"Days to expiry: **{user_row.get('days_to_expiry', 'N/A')}**")
            st.write(f"Auto renew: **{'ON' if user_row.get('current_auto_renew') == 1 else 'OFF'}**")
            st.write(f"Total transactions: **{user_row.get('total_transactions', 'N/A')}**")
            st.write(f"Cancel rate: **{user_row.get('cancel_rate', 0):.1%}**")
            st.write(f"Churned last month: **{'Yes' if user_row.get('last_churn') == 1 else 'No'}**")
        with col_b:
            st.markdown("**Listening Signals**")
            days_listen = user_row.get("days_since_last_listen")
            st.write(f"Days since last listen: **{days_listen if pd.notna(days_listen) else 'Never listened'}**")
            st.write(f"Active days (30d): **{user_row.get('active_days_30d', 0):.0f}**")
            st.write(f"Listening trend: **{user_row.get('listening_trend_ratio', 0):.2f}x**")
            st.write(f"Completion rate (15d): **{user_row.get('completion_rate_15d', 0):.1%}**")
            st.write(f"Stopped listening: **{'Yes' if user_row.get('stopped_listening') == 1 else 'No'}**")
        with col_c:
            st.markdown("**Value Signals**")
            st.write(f"Avg amount paid: **{user_row.get('avg_amount_paid', 0):.0f} TWD**")
            st.write(f"Customer tenure: **{user_row.get('customer_tenure_days', 0):.0f} days**")
            st.write(f"Auto renew rate: **{user_row.get('auto_renew_rate', 0):.1%}**")
            st.write(f"Total paid: **{user_row.get('total_amount_paid', 0):,.0f} TWD**")
            st.write(f"Pay per day: **{user_row.get('pay_per_day', 0):.2f} TWD**")

        st.divider()
        st.subheader("💡 Recommended Action")
        st.info(action)

        st.subheader("Key Risk Drivers")
        drivers = []
        if user_row.get("days_since_last_transaction", 0) > 30:
            drivers.append(f"No transaction in {user_row.get('days_since_last_transaction'):.0f} days")
        if user_row.get("current_auto_renew") == 0:
            drivers.append("Auto-renew is OFF")
        if user_row.get("last_churn") == 1:
            drivers.append("Churned last month — 67.7% re-churn rate")
        if user_row.get("stopped_listening") == 1:
            drivers.append("Stopped listening in March")
        if user_row.get("days_to_expiry", 30) < 7:
            drivers.append(f"Membership expires in {user_row.get('days_to_expiry'):.0f} days")
        if user_row.get("cancel_rate", 0) > 0.3:
            drivers.append(f"High cancellation rate: {user_row.get('cancel_rate'):.1%}")

        if drivers:
            for d in drivers:
                st.write(f"- {d}")
        else:
            st.write("No major risk signals detected.")

        st.divider()
        st.subheader("AI Retention Analyst")
        if st.button("Generate AI Insights", type="primary"):
            with st.spinner("Analyzing user profile..."):
                user_features_dict = {
                    "days_since_last_transaction": user_row.get("days_since_last_transaction", 0),
                    "days_since_last_listen":      user_row.get("days_since_last_listen", 999) if pd.notna(user_row.get("days_since_last_listen")) else 999,
                    "current_auto_renew":          user_row.get("current_auto_renew", 1),
                    "customer_tenure_days":        user_row.get("customer_tenure_days", 0),
                    "last_churn":                  user_row.get("last_churn", 0)
                }
                ai_response = get_ai_insights(
                    churn_prob, segment, drivers, clv, user_features_dict
                )
            st.write(ai_response)
