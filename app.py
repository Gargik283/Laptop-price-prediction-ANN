import os

# Reduce TensorFlow startup messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
# MODEL SETTINGS
# Matching your training notebook
# ============================================================

CATEGORICAL_COLUMNS = [
    "Company",
    "TypeName",
    "ScreenResolution",
    "Cpu",
    "Memory",
    "Gpu",
    "OpSys"
]

NUMERICAL_COLUMNS = [
    "Inches",
    "Ram",
    "Weight"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "laptop_price_model.keras"
    )

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    return model, scaler, feature_columns


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    try:
        return pd.read_csv(
            "laptop_price.csv",
            encoding="latin-1"
        )

    except Exception:
        return None


# ============================================================
# LOAD FILES
# ============================================================

try:

    model, scaler, feature_columns = load_model()

except Exception as e:

    st.error("Model files could not be loaded.")

    st.write(
        "Make sure these files are in the same folder as app.py:"
    )

    st.code(
        """
laptop_price_model.keras
scaler.pkl
feature_columns.pkl
"""
    )

    st.stop()


df = load_dataset()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💻 Laptop Price")

    st.caption("Machine Learning Prediction App")

    st.divider()

    st.subheader("About")

    st.write(
        """
        This application predicts the estimated price
        of a laptop from its technical specifications.
        """
    )

    st.divider()

    st.subheader("Model")

    st.write("**Algorithm:** Artificial Neural Network")
    st.write("**Task:** Regression")
    st.write("**Target:** Price_euros")
    st.write("**Currency:** Euros")

    st.divider()

    st.subheader("Features")

    st.write(
        """
        • Company  
        • Type Name  
        • Screen Size  
        • Screen Resolution  
        • CPU  
        • RAM  
        • Memory  
        • GPU  
        • Operating System  
        • Weight
        """
    )

    st.divider()

    st.caption(
        "Built with Python, TensorFlow and Streamlit"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title("💻 Laptop Price Predictor")

st.write(
    "Estimate the price of a laptop based on its "
    "hardware and software specifications."
)

st.divider()


# ============================================================
# INTRODUCTION
# ============================================================

st.subheader("📋 Enter Laptop Specifications")

st.info(
    "Select the specifications below and click "
    "**Predict Laptop Price** to generate an estimate."
)


# ============================================================
# DATASET OPTIONS
# ============================================================

def get_options(column, default_values):

    if df is not None and column in df.columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        values.sort()

        if values:
            return values

    return default_values


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
        "1366x768",
        "Full HD 1920x1080"
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
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2)

with col1:

    company = st.selectbox(
        "🏢 Company",
        company_options
    )

with col2:

    laptop_type = st.selectbox(
        "💼 Laptop Type",
        type_options
    )


col1, col2, col3 = st.columns(3)

with col1:

    inches = st.number_input(
        "📏 Screen Size (Inches)",
        min_value=10.0,
        max_value=25.0,
        value=15.6,
        step=0.1
    )

with col2:

    ram = st.number_input(
        "🧠 RAM (GB)",
        min_value=2,
        max_value=64,
        value=8,
        step=2
    )

with col3:

    weight = st.number_input(
        "⚖️ Weight (kg)",
        min_value=0.5,
        max_value=6.0,
        value=2.0,
        step=0.1
    )


col1, col2 = st.columns(2)

with col1:

    resolution = st.selectbox(
        "🖥️ Screen Resolution",
        resolution_options
    )

with col2:

    operating_system = st.selectbox(
        "⚙️ Operating System",
        os_options
    )


col1, col2 = st.columns(2)

with col1:

    cpu = st.selectbox(
        "🔧 Processor (CPU)",
        cpu_options
    )

with col2:

    memory = st.selectbox(
        "💾 Storage",
        memory_options
    )


gpu = st.selectbox(
    "🎮 Graphics Card (GPU)",
    gpu_options
)


# ============================================================
# PREDICT BUTTON
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
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Company": [company],

            "TypeName": [laptop_type],

            "Inches": [inches],

            "ScreenResolution": [resolution],

            "Cpu": [cpu],

            "Ram": [ram],

            "Memory": [memory],

            "Gpu": [gpu],

            "OpSys": [operating_system],

            "Weight": [weight]

        })


        # ----------------------------------------------------
        # ONE-HOT ENCODING
        #
        # Same approach used during training
        # ----------------------------------------------------

        encoded_data = pd.get_dummies(
            input_data,
            columns=CATEGORICAL_COLUMNS,
            drop_first=True
        )


        # ----------------------------------------------------
        # MATCH EXACT TRAINING FEATURES
        # ----------------------------------------------------

        encoded_data = encoded_data.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # SCALE NUMERICAL FEATURES
        # ----------------------------------------------------

        encoded_data[NUMERICAL_COLUMNS] = scaler.transform(
            encoded_data[NUMERICAL_COLUMNS]
        )


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        model_input = encoded_data.astype(
            np.float32
        ).to_numpy()


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            model_input,
            verbose=0
        )


        predicted_price = float(
            prediction.flatten()[0]
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success("Prediction completed successfully!")

        st.subheader("💰 Estimated Laptop Price")

        st.metric(
            label="Predicted Price",
            value=f"€{predicted_price:,.2f}"
        )


        # ----------------------------------------------------
        # SPECIFICATION SUMMARY
        # ----------------------------------------------------

        st.subheader("📊 Laptop Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.metric(
                "Company",
                company
            )

        with summary_col2:

            st.metric(
                "RAM",
                f"{ram} GB"
            )

        with summary_col3:

            st.metric(
                "Weight",
                f"{weight} kg"
            )


        with st.expander(
            "🔍 View Complete Specifications"
        ):

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
                    company,
                    laptop_type,
                    f"{inches} inches",
                    resolution,
                    cpu,
                    f"{ram} GB",
                    memory,
                    gpu,
                    operating_system,
                    f"{weight} kg"
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.write(
            "Technical details:"
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Laptop Price Prediction • Artificial Neural Network • "
    "TensorFlow • Streamlit"
)
import os

# Reduce TensorFlow startup messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
# MODEL SETTINGS
# Matching your training notebook
# ============================================================

CATEGORICAL_COLUMNS = [
    "Company",
    "TypeName",
    "ScreenResolution",
    "Cpu",
    "Memory",
    "Gpu",
    "OpSys"
]

NUMERICAL_COLUMNS = [
    "Inches",
    "Ram",
    "Weight"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "laptop_price_model.keras"
    )

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    return model, scaler, feature_columns


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    try:
        return pd.read_csv(
            "laptop_price.csv",
            encoding="latin-1"
        )

    except Exception:
        return None


# ============================================================
# LOAD FILES
# ============================================================

try:

    model, scaler, feature_columns = load_model()

except Exception as e:

    st.error("Model files could not be loaded.")

    st.write(
        "Make sure these files are in the same folder as app.py:"
    )

    st.code(
        """
laptop_price_model.keras
scaler.pkl
feature_columns.pkl
"""
    )

    st.stop()


df = load_dataset()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💻 Laptop Price")

    st.caption("Machine Learning Prediction App")

    st.divider()

    st.subheader("About")

    st.write(
        """
        This application predicts the estimated price
        of a laptop from its technical specifications.
        """
    )

    st.divider()

    st.subheader("Model")

    st.write("**Algorithm:** Artificial Neural Network")
    st.write("**Task:** Regression")
    st.write("**Target:** Price_euros")
    st.write("**Currency:** Euros")

    st.divider()

    st.subheader("Features")

    st.write(
        """
        • Company  
        • Type Name  
        • Screen Size  
        • Screen Resolution  
        • CPU  
        • RAM  
        • Memory  
        • GPU  
        • Operating System  
        • Weight
        """
    )

    st.divider()

    st.caption(
        "Built with Python, TensorFlow and Streamlit"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title("💻 Laptop Price Predictor")

st.write(
    "Estimate the price of a laptop based on its "
    "hardware and software specifications."
)

st.divider()


# ============================================================
# INTRODUCTION
# ============================================================

st.subheader("📋 Enter Laptop Specifications")

st.info(
    "Select the specifications below and click "
    "**Predict Laptop Price** to generate an estimate."
)


# ============================================================
# DATASET OPTIONS
# ============================================================

def get_options(column, default_values):

    if df is not None and column in df.columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        values.sort()

        if values:
            return values

    return default_values


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
        "1366x768",
        "Full HD 1920x1080"
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
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2)

with col1:

    company = st.selectbox(
        "🏢 Company",
        company_options
    )

with col2:

    laptop_type = st.selectbox(
        "💼 Laptop Type",
        type_options
    )


col1, col2, col3 = st.columns(3)

with col1:

    inches = st.number_input(
        "📏 Screen Size (Inches)",
        min_value=10.0,
        max_value=25.0,
        value=15.6,
        step=0.1
    )

with col2:

    ram = st.number_input(
        "🧠 RAM (GB)",
        min_value=2,
        max_value=64,
        value=8,
        step=2
    )

with col3:

    weight = st.number_input(
        "⚖️ Weight (kg)",
        min_value=0.5,
        max_value=6.0,
        value=2.0,
        step=0.1
    )


col1, col2 = st.columns(2)

with col1:

    resolution = st.selectbox(
        "🖥️ Screen Resolution",
        resolution_options
    )

with col2:

    operating_system = st.selectbox(
        "⚙️ Operating System",
        os_options
    )


col1, col2 = st.columns(2)

with col1:

    cpu = st.selectbox(
        "🔧 Processor (CPU)",
        cpu_options
    )

with col2:

    memory = st.selectbox(
        "💾 Storage",
        memory_options
    )


gpu = st.selectbox(
    "🎮 Graphics Card (GPU)",
    gpu_options
)


# ============================================================
# PREDICT BUTTON
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
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Company": [company],

            "TypeName": [laptop_type],

            "Inches": [inches],

            "ScreenResolution": [resolution],

            "Cpu": [cpu],

            "Ram": [ram],

            "Memory": [memory],

            "Gpu": [gpu],

            "OpSys": [operating_system],

            "Weight": [weight]

        })


        # ----------------------------------------------------
        # ONE-HOT ENCODING
        #
        # Same approach used during training
        # ----------------------------------------------------

        encoded_data = pd.get_dummies(
            input_data,
            columns=CATEGORICAL_COLUMNS,
            drop_first=True
        )


        # ----------------------------------------------------
        # MATCH EXACT TRAINING FEATURES
        # ----------------------------------------------------

        encoded_data = encoded_data.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # SCALE NUMERICAL FEATURES
        # ----------------------------------------------------

        encoded_data[NUMERICAL_COLUMNS] = scaler.transform(
            encoded_data[NUMERICAL_COLUMNS]
        )


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        model_input = encoded_data.astype(
            np.float32
        ).to_numpy()


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            model_input,
            verbose=0
        )


        predicted_price = float(
            prediction.flatten()[0]
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success("Prediction completed successfully!")

        st.subheader("💰 Estimated Laptop Price")

        st.metric(
            label="Predicted Price",
            value=f"€{predicted_price:,.2f}"
        )


        # ----------------------------------------------------
        # SPECIFICATION SUMMARY
        # ----------------------------------------------------

        st.subheader("📊 Laptop Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.metric(
                "Company",
                company
            )

        with summary_col2:

            st.metric(
                "RAM",
                f"{ram} GB"
            )

        with summary_col3:

            st.metric(
                "Weight",
                f"{weight} kg"
            )


        with st.expander(
            "🔍 View Complete Specifications"
        ):

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
                    company,
                    laptop_type,
                    f"{inches} inches",
                    resolution,
                    cpu,
                    f"{ram} GB",
                    memory,
                    gpu,
                    operating_system,
                    f"{weight} kg"
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.write(
            "Technical details:"
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Laptop Price Prediction • Artificial Neural Network • "
    "TensorFlow • Streamlit"
)