import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🔮",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipe      = joblib.load("elbehiry_v3_final.pkl")
    threshold = float(np.load("elbehiry_v3_threshold.npy"))
    return pipe, threshold

pipe, threshold = load_model()

# ── Feature engineering (must match training notebook) ────────────────────────
def engineer_features(df):
    freq = df["Frequency"] + 1e-3

    df["Revenue_per_txn"]      = df["Monetary_Value"]  / freq
    df["SupportCalls_per_txn"] = df["Support_Calls"]   / freq
    df["Recency_per_txn"]      = df["Recency_Days"]    / freq
    df["Tenure_per_txn"]       = df["Tenure_Months"]   / freq

    df["Support_x_Recency"]   = df["Support_Calls"]  * df["Recency_Days"]
    df["Revenue_x_Tenure"]    = df["Monetary_Value"] * df["Tenure_Months"]
    df["Support_x_LowTenure"] = df["Support_Calls"]  * (1 / (df["Tenure_Months"] + 1))

    df["HighSupportCalls"]            = (df["Support_Calls"] >= 7).astype(int)
    df["LowTenure"]                   = (df["Tenure_Months"] <= 6).astype(int)
    df["HighRecency"]                 = (df["Recency_Days"]  >= 300).astype(int)
    df["LowFrequency"]                = (df["Frequency"]     <= 10).astype(int)
    df["HighSupport_AND_HighRecency"]  = (
        (df["Support_Calls"] >= 7) & (df["Recency_Days"] >= 300)
    ).astype(int)
    return df

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔮 Customer Churn Prediction")
st.markdown("Fill in the customer details below to predict whether they are likely to churn.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")
    age    = st.number_input("Age",             min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender",             ["Male", "Female"])
    tenure = st.number_input("Tenure (Months)", min_value=0,  max_value=120, value=12)

with col2:
    st.subheader("📊 Behaviour Metrics")
    recency   = st.number_input("Recency (Days since last purchase)", min_value=0,   max_value=730,      value=90)
    frequency = st.number_input("Frequency (Number of transactions)", min_value=1,   max_value=500,      value=20)
    monetary  = st.number_input("Monetary Value (Total spend $)",     min_value=0.0, max_value=100000.0, value=500.0, step=10.0)
    support   = st.number_input("Support Calls",                      min_value=0,   max_value=20,       value=2)

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    input_dict = {
        "Age":           age,
        "Gender":        gender,
        "Tenure_Months": tenure,
        "Recency_Days":  recency,
        "Frequency":     frequency,
        "Monetary_Value": monetary,
        "Support_Calls": support,
    }
    df_input = pd.DataFrame([input_dict])
    df_input = engineer_features(df_input)

    proba      = pipe.predict_proba(df_input)[:, 1][0]
    will_churn = proba >= threshold

    st.divider()
    st.subheader("📋 Prediction Result")

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        if will_churn:
            st.error("⚠️ **WILL CHURN**")
        else:
            st.success("✅ **RETAINED**")

    with col_res2:
        st.metric("Churn Probability", f"{proba*100:.1f}%")

    with col_res3:
        st.metric("Threshold Used", f"{threshold:.3f}")

    # ── Risk gauge ────────────────────────────────────────────────────────────
    st.markdown("#### Risk Level")
    bar_color  = "🔴" if proba >= 0.7 else ("🟡" if proba >= 0.4 else "🟢")
    risk_label = "High Risk" if proba >= 0.7 else ("Medium Risk" if proba >= 0.4 else "Low Risk")
    st.progress(float(proba), text=f"{bar_color} {risk_label} — {proba*100:.1f}%")

    # ── Risk factors ──────────────────────────────────────────────────────────
    st.markdown("#### ⚡ Key Risk Factors Detected")
    flags = []
    if support >= 7:
        flags.append("🔴 High support calls (≥7) — strong churn signal")
    if recency >= 300:
        flags.append("🔴 High recency (≥300 days) — customer is inactive")
    if tenure <= 6:
        flags.append("🟡 Low tenure (≤6 months) — new customer, higher churn risk")
    if frequency <= 10:
        flags.append("🟡 Low transaction frequency — low engagement")
    if support >= 7 and recency >= 300:
        flags.append("🔴 CRITICAL: High support calls + high recency combined")

    if flags:
        for f in flags:
            st.markdown(f"- {f}")
    else:
        st.markdown("- 🟢 No major risk factors detected")

    # ── Recommendation ────────────────────────────────────────────────────────
    st.markdown("#### 💡 Recommended Action")
    if proba >= 0.7:
        st.warning("**Immediate action needed.** Assign a retention agent, offer a personalised discount or loyalty reward.")
    elif proba >= 0.4:
        st.info("**Monitor closely.** Send a re-engagement email or a small incentive.")
    else:
        st.success("**No action needed.** Customer appears loyal and engaged.")

# ── Batch prediction ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📁 Batch Prediction (CSV Upload)")
st.markdown("Upload a CSV with columns: `Age, Gender, Tenure_Months, Recency_Days, Frequency, Monetary_Value, Support_Calls`")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    batch_fe = engineer_features(batch_df.copy())
    probas   = pipe.predict_proba(batch_fe)[:, 1]
    preds    = (probas >= threshold).astype(int)

    batch_df["Churn_Probability"] = (probas * 100).round(1)
    batch_df["Predicted_Churn"]   = preds
    batch_df["Risk"]              = pd.cut(
        probas,
        bins=[0, 0.4, 0.7, 1.0],
        labels=["🟢 Low", "🟡 Medium", "🔴 High"]
    )

    st.success(f"✅ Processed {len(batch_df)} customers — "
               f"{preds.sum()} predicted to churn ({preds.mean()*100:.1f}%)")

    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Total Customers",   len(batch_df))
    col_b2.metric("Predicted Churners", preds.sum())
    col_b3.metric("Churn Rate",         f"{preds.mean()*100:.1f}%")

    st.dataframe(batch_df.sort_values("Churn_Probability", ascending=False),
                 use_container_width=True)

    csv = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", csv,
                       file_name="churn_predictions.csv", mime="text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: LogisticRegression with SMOTE | Recall: 0.90 | AUC: 0.937 | Threshold: optimised for 90% churn recall")
