import os
import sys

# Suppress background TensorFlow logs and startup text strings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# ============================================================
# PAGE ARCHITECTURE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Laptop Price Predictor ANN",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom css style mapping for layout presentation visibility
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2.25rem !important; font-weight: 700; color: #1E88E5; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# RUNTIME ASSET VERIFICATION LOGIC (Diagnostic Overlay)
# ============================================================
required_assets = ["laptop_price_model.keras", "scaler.pkl", "feature_columns.pkl", "laptop_price.csv"]
missing_assets = []

# Scan directory structures to evaluate physical presence
for asset in required_assets:
    # Check current directory and subdirectories for case-insensitive matches
    found = False
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.lower() == asset.lower():
                found = True
                # Correct the internal naming reference mapping if path is relative
                actual_path = os.path.join(root, f).replace(".\\", "").replace("./", "")
                if asset == "laptop_price_model.keras": model_path = actual_path
                elif asset == "scaler.pkl": scaler_path = actual_path
                elif asset == "feature_columns.pkl": feature_columns_path = actual_path
                elif asset == "laptop_price.csv": dataset_path = actual_path
                break
    if not found:
        missing_assets.append(asset)

# Intercept broken layouts before execution fails
if missing_assets:
    st.error("🚨 Critical Error: Missing Machine Learning Pipeline Files")
    st.write(f"The cloud container scanned the server storage path `{os.getcwd()}` but cannot locate these necessary assets:")
    for missing in missing_assets:
        st.markdown(f"* **`{missing}`**")
    
    st.info("💡 **How to Fix:** Ensure these exact files are uploaded directly alongside your code in your root repository.")
    st.stop()


# ============================================================
# RESOURCE CACHING & DATA PIPELINE LOADING
# ============================================================
@st.cache_resource
def load_ml_assets(m_p, s_p, f_p):
    """Loads and caches the compiled deep learning assets dynamically using tracked file paths."""
    model = tf.keras.models.load_model(m_p)
    with open(s_p, "rb") as f: scaler = pickle.load(f)
    with open(f_p, "rb") as f: feature_columns = pickle.load(f)
    return model, scaler, feature_columns

@st.cache_data
def load_clean_dataset(d_p):
    """Loads dataset and strips away unexpected structural noise elements."""
    try:
        raw_df = pd.read_csv(d_p, encoding="latin-1")
        if 'laptop_ID' in raw_df.columns:
            corrupted = raw_df['laptop_ID'].astype(str).str.contains('<<<<<<<|=======|>>>>>>>')
            return raw_df[~corrupted].copy()
        return raw_df
    except:
        # Emergency array fallbacks if storage streaming is interrupted
        return pd.DataFrame({
            "Company": ["Acer", "Apple", "Asus", "Dell", "HP", "Lenovo", "MSI"],
            "TypeName": ["Notebook", "Ultrabook", "Gaming", "2 in 1 Convertible", "Workstation"],
            "ScreenResolution": ["Full HD 1920x1080", "1366x768"],
            "Cpu": ["Intel Core i5 7200U 2.5GHz"], "Memory": ["256GB SSD", "512GB SSD"],
            "Gpu": ["Intel HD Graphics 620"], "OpSys": ["Windows 10", "macOS", "Linux"]
        })

# Instantiate pipeline layers safely using our verified paths
model, scaler, feature_columns = load_ml_assets(model_path, scaler_path, feature_columns_path)
df = load_clean_dataset(dataset_path)

CATEGORICAL_COLUMNS = ["Company", "TypeName", "ScreenResolution", "Cpu", "Memory", "Gpu", "OpSys"]
NUMERICAL_COLUMNS = ["Inches", "Ram", "Weight"]

def extract_dropdown_options(column_name, fallback_list):
    if df is not None and column_name in df.columns:
        options = df[column_name].dropna().astype(str).unique().tolist()
        options.sort()
        return options
    return fallback_list


# ============================================================
# USER INTERFACE RENDERING
# ============================================================
with st.sidebar:
    st.title("💻 Architecture Summary")
    st.markdown("""
    ### Model Topology
    * **Type:** Artificial Neural Network (ANN)
    * **Task:** Continuous Price Mappings
    * **Input Vector Matrix:** 337 Features
    """)

st.title("💻 Laptop Price Prediction via Deep Learning")
st.write("Specify configuration parameters below to generate evaluations via the underlying Neural Network grid.")
st.divider()

st.subheader("📋 Step 1: Core System Dimensions")
metric_grid1, metric_grid2, metric_grid3 = st.columns(3)
with metric_grid1: user_inches = st.number_input("📏 Screen Dimension (Inches)", 10.0, 25.0, 15.6, 0.1)
with metric_grid2: user_ram = st.number_input("🧠 System RAM Configuration (GB)", 2, 64, 8, 2)
with metric_grid3: user_weight = st.number_input("⚖️ Machine Weight (kg)", 0.5, 6.0, 2.0, 0.1)

st.divider()
st.subheader("⚙️ Step 2: Component Specifications")
ui_col1, ui_col2 = st.columns(2)
with ui_col1:
    user_company = st.selectbox("Company", extract_dropdown_options("Company", ["Apple", "HP", "Dell"]))
    user_resolution = st.selectbox("Resolution Matrix", extract_dropdown_options("ScreenResolution", ["Full HD 1920x1080"]))
    user_cpu = st.selectbox("Processor (CPU)", extract_dropdown_options("Cpu", ["Intel Core i5 7200U 2.5GHz"]))
with ui_col2:
    user_typename = st.selectbox("Classification (TypeName)", extract_dropdown_options("TypeName", ["Notebook", "Ultrabook"]))
    user_opsys = st.selectbox("Operating System (OpSys)", extract_dropdown_options("OpSys", ["Windows 10", "macOS"]))
    user_memory = st.selectbox("Storage Module (Memory)", extract_dropdown_options("Memory", ["256GB SSD"]))

user_gpu = st.selectbox("Graphics Card (GPU)", extract_dropdown_options("Gpu", ["Intel HD Graphics 620"]))


# ============================================================
# DEEP LEARNING INFERENCE RUNTIME
# ============================================================
st.divider()
trigger_inference = st.button("🔮 Calculate Estimated Valuation", type="primary", use_container_width=True)

if trigger_inference:
    try:
        raw_input_row = pd.DataFrame({
            "Company": [user_company], "TypeName": [user_typename], "Inches": [user_inches],
            "ScreenResolution": [user_resolution], "Cpu": [user_cpu], "Ram": [user_ram],
            "Memory": [user_memory], "Gpu": [user_gpu], "OpSys": [user_opsys], "Weight": [user_weight]
        })

        processed_row = pd.get_dummies(raw_input_row, columns=CATEGORICAL_COLUMNS, drop_first=True)
        aligned_features = processed_row.reindex(columns=feature_columns, fill_value=0)
        aligned_features[NUMERICAL_COLUMNS] = scaler.transform(aligned_features[NUMERICAL_COLUMNS])
        
        network_vector = aligned_features.astype(np.float32).to_numpy()
        predicted_tensor = model.predict(network_vector, verbose=0)
        final_euros_value = max(0.0, float(predicted_tensor.flatten()[0]))

        st.success("🎉 Neural Network Estimation Completed Successfully!")
        display_col1, display_col2 = st.columns(2)
        with display_col1: st.metric(label="Predicted Valuation (Euros)", value=f"€{final_euros_value:,.2f}")
        with display_col2: st.metric(label="Approximate Valuation (INR)", value=f"₹{final_euros_value * 91.5:,.2f}")

    except Exception as error_exception:
        st.error("🚨 Inference Evaluation Failure.")
        st.exception(error_exception)
