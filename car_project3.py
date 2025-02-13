import streamlit as st
import pandas as pd
import pickle
import numpy as np
import base64
from scipy import stats
from streamlit_extras.colored_header import colored_header

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Model and Encoders
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    
    base64_image = base64.b64encode(data).decode("utf-8")
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Change all text to white */
        html, body, [class*="st-"] {{
            color: white !important;
        }}

        /* Style for prediction result box */
        .prediction-box {{
            background-color: rgba(0, 255, 0, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            color: white !important;  /* Ensure text inside box is white */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set Background Image
set_background_image_local("E:\Project1\Dark-bg.jpg")  # Update with correct path

# Load Model
car_model = load_model("E:/Project1/carmodel1.pkl")

# Load Dataset
df = pd.read_csv("E:/Project1/car_price_ai.csv")

# Load Encoders
encoders = {
    "brand": load_model("E:/Project1/encoder_brand.pkl"),
    "fuel_type": load_model("E:/Project1/encoder_fuel_type.pkl"),
    "insurance_type": load_model("E:/Project1/encoder_insurance_type.pkl"),
    "location": load_model("E:/Project1/encoder_location.pkl"),
    "no_of_seats": load_model("E:/Project1/encoder_no_of_seats.pkl"),
    "ownership": load_model("E:/Project1/encoder_ownership.pkl"),
    "transmission": load_model("E:/Project1/encoder_transmission.pkl"),
    "model": load_model("E:/Project1/encoder_model.pkl")
}

# Title & Tabs
st.markdown("<h1 style='text-align: center; color: #FFA500;'>🚗 Car Price Prediction</h1>", unsafe_allow_html=True)
colored_header(label="🔹 Enter Car Details", description="Provide details to predict the price.", color_name="orange-70")
# User Inputs - Organized in Columns
col1, col2, col3 = st.columns(3)
with col1:
    brand = st.selectbox('Brand', df["brand"].unique())
    brand = encoders["brand"].transform([[brand]])[0][0]
    model = st.selectbox('Model', df["model"].unique())
    model = encoders["model"].transform([[model]])[0][0]
    fuel_type = st.selectbox('Fuel Type', df["fuel_type"].unique())
    fuel_type = encoders["fuel_type"].transform([[fuel_type]])[0][0]
    transmission = st.selectbox('Transmission', df["transmission"].unique())
    transmission = encoders["transmission"].transform([[transmission]])[0][0]
with col2:
    insurance_type = st.selectbox('Insurance Type', df["insurance_type"].unique())
    insurance_type = encoders["insurance_type"].transform([[insurance_type]])[0][0]
    no_of_seats = st.selectbox('Number of Seats', df["no_of_seats"].unique())
    no_of_seats = encoders["no_of_seats"].transform([[no_of_seats]])[0][0]
    ownership = st.selectbox('Ownership', df["ownership"].unique())
    ownership = encoders["ownership"].transform([[ownership]])[0][0]
    location = st.selectbox('Location', df["location"].unique())
    location = encoders["location"].transform([[location]])[0][0]
with col3:
    km_driven_data = st.number_input("KM Driven", min_value=10000, step=500)
    make_year = st.number_input("Make Year", min_value=1990, max_value=2025, step=1)
    registration_year = st.number_input("Registration Year", min_value=1990, max_value=2025, step=1)
    power_data = st.number_input("Power (BHP)", min_value=100.0, step=5.0)
# Additional Inputs
col4, col5 = st.columns(2)
with col4:
    cc = st.number_input("Engine CC", min_value=998.0, step=100.0)
with col5:
    mileage = st.number_input("Mileage (kmpl)", min_value=18.0, step=1.0)
# Normalize Skewed Data
km_driven = np.cbrt(km_driven_data)  # Cube root transformation
power_transformed = stats.boxcox([power_data], lmbda=-0.5)[0]  # Box-Cox transformation
# Create DataFrame for Model
input_data = pd.DataFrame({
    'location': [location],
    'make_year': [int(make_year)],
    'brand': [brand],
    'model': [model],
    'km_driven': [float(km_driven)],
    'fuel_type': [fuel_type],
    'transmission': [transmission],
    'registration_year': [int(registration_year)],
    "insurance_type": [insurance_type],
    'no_of_seats': [int(no_of_seats)],
    'ownership': [ownership],
    'power': [float(power_transformed)],  # Transformed power
    'cc': [float(cc)],
    'mileage': [float(mileage)]
})
# Predict Button
st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
if st.button("🚀 Predict Price"):
    prediction = car_model.predict(input_data)
    predicted_price = np.exp(prediction[0])  # Inverse log transformation
    st.markdown(
        f"<div class='prediction-box'>💰 <b>Estimated Price:</b> Rs {int(predicted_price):,}</div>",
        unsafe_allow_html=True
    )




mlflow.set_experiment("Car_Price_Prediction")

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": xgb.XGBRegressor(),
    "SVR": SVR()
}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Calculate Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("Model", model_name)
        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("R² Score", r2)

        # Log the model
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name}:")
        print(f"  Mean Squared Error: {mse:.2f}")
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  R² Score: {r2:.2f}\n")

print("All models trained and logged in MLflow.")