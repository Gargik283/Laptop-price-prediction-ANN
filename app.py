import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# [Code omitted for brevity. Please refer to the original code structure]
# The code sets up a Streamlit app to predict laptop prices using TensorFlow.
# It includes model loading (with caching), input components (selectboxes/number inputs),
# data preprocessing, and prediction display.

# --- REPLACE YOUR ENTIRE FILE WITH THIS CLEANED VERSION ---
# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("laptop_price_model.keras")
    with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f: feature_columns = pickle.load(f)
    return model, scaler, feature_columns

@st.cache_data
def load_dataset():
    try: return pd.read_csv("laptop_price.csv", encoding="latin-1")
    except: return None

# Load resources
try:
    model, scaler, feature_columns = load_model()
    df = load_dataset()
except Exception as e:
    st.error("Error loading model files. Ensure .keras and .pkl files are present.")
    st.stop()

# ============================================================
# APP UI
# ============================================================
st.title("💻 Laptop Price Predictor")
# ... [Rest of the UI and prediction logic follows, ensuring no duplicates]
