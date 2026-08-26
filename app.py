import os

# Suppress TensorFlow startup messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "laptop_price_model.keras"
SCALER_FILE = "scaler.pkl"
FEATURE_COLUMNS_FILE = "feature_columns.pkl"
DATASET_FILE = "laptop_price.csv"


# ============================================================
# FEATURES
# Based on the training notebook
# ============================================================

CATEGORICAL_FEATURES = [
    "Company",
    "TypeName",
    "ScreenResolution",
    "Cpu",
    "Memory",
    "Gpu",
    "OpSys"
]

NUMERICAL_FEATURES = [
    "Inches",
    "Ram",
    "Weight"
]


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_COLUMNS_FILE,
    DATASET_FILE
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("❌ Required project files are missing.")

    st.write("Please make sure these files are in the same folder as `app.py`:")

    for file in missing_files:
        st.code(file)

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_assets():

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    with open(SCALER_FILE, "rb") as file:
        scaler = pickle.load(file)

    with open(FEATURE_COLUMNS_FILE, "rb") as file:
        feature_columns = pickle.load(file)

    return model, scaler, feature_columns


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        DATASET_FILE,
        encoding="latin-1"
    )


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    model, scaler, feature_columns = load_model_assets()
    df = load_dataset()

except Exception as error:

    st.error("❌ Unable to load the trained model or preprocessing files.")

    st.exception(error)

    st.stop()


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_options(column_name, fallback_values):

    if column_name in df.columns:

        values = (
            df[column_name]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        values.sort()

        if values:
            return values

    return fallback_values


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💻 Laptop Price Predictor")

    st.caption(
        "Deep Learning Regression Application"
    )

    st.divider()

    st.subheader("🧠 Model Information")

    st.write("**Approach:** Deep Learning")
    st.write("**Architecture:** Artificial Neural Network")
    st.write("**Task:** Regression")
    st.write("**Target:** Price_euros")
    st.write(
        f"**Encoded Inputs:** {len(feature_columns)}"
    )
    st.write("**Trainable Parameters:** 70,145")

    st.divider()

    st.subheader("🛠️ Technologies")

    st.write(
        """
        • Python  
        • Pandas  
        • NumPy  
        • Scikit-learn  
        • TensorFlow  
        • Keras  
        • Streamlit
        """
    )

    st.divider()

    st.subheader("📊 Model Performance")

    st.write("**R²:** 0.8428")
    st.write("**MAE:** €186.09")

    st.divider()

    st.caption(
        "End-to-end Deep Learning project: "
        "preprocessing → ANN training → deployment"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("💻 Laptop Price Predictor")

st.write(
    "Predict the estimated price of a laptop from "
    "its hardware and software specifications."
)

st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:

    st.metric(
        "Model",
        "ANN"
    )

with overview_col2:

    st.metric(
        "Task",
        "Regression"
    )

with overview_col3:

    st.metric(
        "R² Score",
        "0.8428"
    )


st.divider()


# ============================================================
# LAPTOP SPECIFICATIONS
# ============================================================

st.subheader("📋 Laptop Specifications")

st.caption(
    "Select the laptop configuration below and generate "
    "a price prediction."
)


# ============================================================
# DROPDOWN OPTIONS
# ============================================================

company_options = get_options(
    "Company",
    [
        "Acer",
        "Apple",
        "Asus",
        "Dell",
        "HP",
        "Lenovo"
    ]
)

type_options = get_options(
    "TypeName",
    [
        "Notebook",
        "Ultrabook",
        "Gaming",
        "2 in 1 Convertible",
        "Workstation",
        "Netbook"
    ]
)

resolution_options = get_options(
    "ScreenResolution",
    [
        "Full HD 1920x1080",
        "1366x768"
    ]
)

cpu_options = get_options(
    "Cpu",
    [
        "Intel Core i5 7200U 2.5GHz"
    ]
)

memory_options = get_options(
    "Memory",
    [
        "256GB SSD",
        "512GB SSD",
        "1TB HDD"
    ]
)

gpu_options = get_options(
    "Gpu",
    [
        "Intel HD Graphics 620"
    ]
)

os_options = get_options(
    "OpSys",
    [
        "Windows 10",
        "Windows 7",
        "Linux",
        "macOS"
    ]
)


# ============================================================
# NUMERICAL INPUTS
# ============================================================

st.markdown("### ⚙️ Core Specifications")

col1, col2, col3 = st.columns(3)

with col1:

    user_inches = st.number_input(
        "📏 Screen Size (Inches)",
        min_value=10.1,
        max_value=18.4,
        value=15.6,
        step=0.1
    )

with col2:

    user_ram = st.number_input(
        "🧠 RAM (GB)",
        min_value=2,
        max_value=64,
        value=8,
        step=2
    )

with col3:

    user_weight = st.number_input(
        "⚖️ Weight (kg)",
        min_value=0.5,
        max_value=6.0,
        value=2.0,
        step=0.1
    )


# ============================================================
# CATEGORICAL INPUTS
# ============================================================

st.markdown("### 🧩 Hardware & Software")

col1, col2 = st.columns(2)

with col1:

    user_company = st.selectbox(
        "🏢 Company",
        company_options
    )

    user_typename = st.selectbox(
        "💼 Laptop Type",
        type_options
    )

    user_cpu = st.selectbox(
        "⚙️ Processor (CPU)",
        cpu_options
    )

    user_gpu = st.selectbox(
        "🎮 Graphics Card (GPU)",
        gpu_options
    )

with col2:

    user_resolution = st.selectbox(
        "🖥️ Screen Resolution",
        resolution_options
    )

    user_memory = st.selectbox(
        "💾 Memory / Storage",
        memory_options
    )

    user_opsys = st.selectbox(
        "🪟 Operating System",
        os_options
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Laptop Price",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CREATE RAW INPUT
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Company": [user_company],

            "TypeName": [user_typename],

            "Inches": [user_inches],

            "ScreenResolution": [user_resolution],

            "Cpu": [user_cpu],

            "Ram": [user_ram],

            "Memory": [user_memory],

            "Gpu": [user_gpu],

            "OpSys": [user_opsys],

            "Weight": [user_weight]

        })


        # ----------------------------------------------------
        # ONE-HOT ENCODING
        # Same categorical columns used during training
        # ----------------------------------------------------

        encoded_data = pd.get_dummies(
            input_data,
            columns=CATEGORICAL_FEATURES,
            drop_first=True
        )


        # ----------------------------------------------------
        # ALIGN WITH TRAINING FEATURE ORDER
        # ----------------------------------------------------

        encoded_data = encoded_data.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # SCALE NUMERICAL FEATURES
        # Same features standardized during training
        # ----------------------------------------------------

        encoded_data[NUMERICAL_FEATURES] = scaler.transform(
            encoded_data[NUMERICAL_FEATURES]
        )


        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        model_input = (
            encoded_data
            .astype(np.float32)
            .to_numpy()
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            model_input,
            verbose=0
        )

        predicted_price = max(
            0.0,
            float(np.asarray(prediction).flatten()[0])
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success(
            "✅ Price prediction generated successfully!"
        )

        st.subheader("💰 Estimated Laptop Price")

        st.metric(
            label="Predicted Price",
            value=f"€{predicted_price:,.2f}"
        )


        # ----------------------------------------------------
        # SPECIFICATION SUMMARY
        # ----------------------------------------------------

        st.subheader("📊 Selected Specifications")

        summary = pd.DataFrame({

            "Specification": [
                "Company",
                "Laptop Type",
                "Screen Size",
                "Screen Resolution",
                "CPU",
                "RAM",
                "Memory",
                "GPU",
                "Operating System",
                "Weight"
            ],

            "Selected Value": [
                user_company,
                user_typename,
                f"{user_inches} inches",
                user_resolution,
                user_cpu,
                f"{user_ram} GB",
                user_memory,
                user_gpu,
                user_opsys,
                f"{user_weight} kg"
            ]

        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


    except Exception as error:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Laptop Price Prediction • Deep Learning • "
    "Artificial Neural Network • TensorFlow / Keras • Streamlit"
)
