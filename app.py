import streamlit as st
import pickle
import pandas as pd
import time  # For progress bar effect

# Load trained model
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Expected features based on training
expected_features = ['CLMSEX', 'CLMAGE', 'Claim_Approval_Status', 'Policy_Type',
                     'Driving_Record', 'settlement_ratio', 'CLMINSUR_LOSS', 'SEATBELT_AccidentSeverity']

# Mappings for categorical features (must match training)
policy_type_mapping = {'Comprehensive': 0, 'Third-Party': 1}
driving_record_mapping = {'Clean': 0, 'Minor Offenses': 1, 'Major Offenses': 2}
seatbelt_accident_severity_mapping = {
    'No Seatbelt - Minor': 0,
    'No Seatbelt - Moderate': 1,
    'No Seatbelt - Severe': 2,
    'Seatbelt - Minor': 3,
    'Seatbelt - Moderate': 4,
    'Seatbelt - Severe': 5
}

# Custom CSS for improved UI
st.markdown("""
    <style>
        .stButton>button {
            background-color: #008CBA;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #005f73;
        }
        .stAlert {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.title("🔍 User Info")
st.sidebar.text("Fill in the details below:")
user_name = st.sidebar.text_input("Enter Your Name", value="Guest")

# ---- Main Title ----
st.title("⚖️ Attorney Involvement Prediction")
st.markdown("""
    **Predict whether an attorney is likely to be involved in an insurance claim.**  
    Fill in the details below and click **Submit** to get a prediction.
""")

# ---- Input Fields ----
st.markdown("### 📌 Claim Details")

col1, col2 = st.columns(2)

with col1:
    clmsex = st.selectbox('👤 Claimant Gender', [1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')
    clmage = st.number_input('🎂 Claimant Age', min_value=1, max_value=100, value=30)
    claim_approval_status = st.selectbox('✅ Claim Approval Status', [1, 0],
                                         format_func=lambda x: 'Approved' if x == 1 else 'Denied')
    policy_type = st.selectbox('📄 Policy Type', list(policy_type_mapping.keys()))

with col2:
    driving_record = st.selectbox('🚗 Driving Record', list(driving_record_mapping.keys()))
    seatbelt_accident_severity = st.selectbox('🎗️ Seatbelt & Accident Severity',
                                              list(seatbelt_accident_severity_mapping.keys()))
    claim_amount_requested = st.number_input('💰 Claim Amount Requested', min_value=0.0, value=10000.0)
    settlement_amount = st.number_input('💵 Settlement Amount', min_value=0.0, value=2000.0)

st.markdown("### 🔎 Additional Details")
clminsur = st.selectbox('🛡️ Insured Claimant', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
loss = st.number_input('📉 Financial Loss', min_value=0.0, value=5000.0)

# ---- Prediction Button ----
if st.button("🚀 Submit"):
    # Progress bar simulation
    with st.spinner("Processing..."):
        time.sleep(1.5)  # Simulate loading time

    # Feature Engineering
    settlement_ratio = settlement_amount / claim_amount_requested if claim_amount_requested != 0 else 0
    CLMINSUR_LOSS = clminsur * loss  # Derived feature

    # Convert categorical inputs to numerical
    Policy_Type = policy_type_mapping[policy_type]
    Driving_Record = driving_record_mapping[driving_record]
    SEATBELT_AccidentSeverity = seatbelt_accident_severity_mapping[seatbelt_accident_severity]

    # Prepare input data
    input_data = pd.DataFrame({
        'CLMSEX': [clmsex],
        'CLMAGE': [clmage],
        'Claim_Approval_Status': [claim_approval_status],
        'Policy_Type': [Policy_Type],
        'Driving_Record': [Driving_Record],
        'settlement_ratio': [settlement_ratio],
        'CLMINSUR_LOSS': [CLMINSUR_LOSS],
        'SEATBELT_AccidentSeverity': [SEATBELT_AccidentSeverity]
    })

    # Ensure input columns match trained model
    input_data = input_data[expected_features]

    # Make prediction
    prediction = model.predict(input_data)

    # ---- Display Prediction Result ----
    st.markdown("## 🔮 Prediction Result:")

    if prediction[0] == 1:
        st.success(f"**Attorney involvement is likely!** ⚖️\n\nCarefully review the case.")
    else:
        st.info(f"**Attorney involvement is unlikely.** ✅\n\nThe case seems straightforward.")

else:
    st.info("📌 Please enter the details and click **Submit** to get a prediction.")
