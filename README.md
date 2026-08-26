# 💻 Laptop Price Prediction

A machine learning regression project that predicts laptop prices based on hardware and software specifications.

## 🚀 Live Demo

[Try the Streamlit App](YOUR_STREAMLIT_APP_LINK)

## 📌 Project Overview

This project uses an Artificial Neural Network (ANN) to predict laptop prices from specifications such as company, processor, RAM, storage, GPU, screen size, operating system, and weight.

The trained model is deployed as an interactive Streamlit web application.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit

## ⚙️ Deep Learning Workflow

- Data preprocessing
- Categorical feature encoding
- Numerical feature scaling
- Train-test split
- Artificial Neural Network regression
- Early stopping
- Model evaluation
- Streamlit deployment

## 📊 Input Features

- Company
- Type Name
- Screen Size
- Screen Resolution
- CPU
- RAM
- Memory
- GPU
- Operating System
- Weight

**Target:** `Price_euros`

## 📁 Project Structure

```text
laptop-price-prediction/
│
├── app.py
├── laptop_price.csv
├── laptop_price_model.keras
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md



```

# 💻 Laptop Price Prediction

A Deep Learning project that uses an Artificial Neural Network (ANN) to predict laptop prices from hardware and software specifications.

## 🚀 Live Demo

[Open the Streamlit App](YOUR_STREAMLIT_APP_LINK)

## 📌 Project Overview

This project builds an Artificial Neural Network (ANN) regression model to estimate laptop prices based on specifications such as processor, RAM, storage, GPU, screen size, operating system, and weight.

The trained neural network is deployed as an interactive Streamlit application.

## 🧠 Deep Learning Model

The project uses a fully connected Artificial Neural Network with the following architecture:

```text
Input Layer
    ↓
Dense (128) + ReLU
    ↓
Dropout (0.2)
    ↓
Dense (128) + ReLU
    ↓
Dropout (0.2)
    ↓
Dense (64) + ReLU
    ↓
Dropout (0.2)
    ↓
Dense (32) + ReLU
    ↓
Dropout (0.2)
    ↓
Dense (1) + Linear Output
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Evaluation Metric: Mean Absolute Error (MAE)
Maximum Epochs: 100
Early Stopping: Patience = 5
Trainable Parameters: 70,145
📊 Dataset

The dataset contains 1,303 laptop records with hardware and software specifications.

Input Features
Company
Type Name
Screen Size
Screen Resolution
CPU
RAM
Memory
GPU
Operating System
Weight
Target

Price_euros

laptop_ID and Product were excluded from model training because they are identifiers/product labels rather than predictive model inputs.

⚙️ Data Preprocessing

The preprocessing pipeline includes:

Converting RAM values from strings such as 8GB to numerical values.
Converting laptop weight from values such as 1.35kg to numerical values.
One-hot encoding categorical features.
Removing laptop_ID and Product.
Standardizing Inches, Ram, and Weight using StandardScaler.
Splitting the data into training and testing sets using an 80:20 split.
📈 Model Performance

The model was evaluated on the test set using MSE, MAE, and R².

Metric	Result
Mean Squared Error (MSE)	79,855.56
Mean Absolute Error (MAE)	€186.09
R² Score	0.8428

The R² score indicates that the model explains approximately 84.28% of the variance in laptop prices on the test set.

🛠️ Tech Stack
Python
Pandas
NumPy
Scikit-learn
TensorFlow
Keras
Streamlit
📁 Project Structure
laptop-price-prediction/
│
├── app.py
├── laptop_price.csv
├── laptop_price_model.keras
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md
▶️ Run Locally
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd laptop-price-prediction
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit application
streamlit run app.py
🎯 Key Highlights
Built an end-to-end Deep Learning regression pipeline.
Designed a fully connected Artificial Neural Network using TensorFlow/Keras.
Applied categorical encoding and numerical feature scaling.
Used Dropout and Early Stopping during model training.
Evaluated the model using MSE, MAE, and R².
Deployed the trained model through an interactive Streamlit application.
👩‍💻 Author

Gargi Kundu

B.Tech – Electronics & Communication Engineering (VLSI Design)

GitHub Profile


---
