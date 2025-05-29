import streamlit as st
import pandas as pd
import pickle
import numpy as np
import base64
from scipy import stats
from streamlit_extras.colored_header import colored_header

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
        
        html, body, [class*="st-"] {{
            color: white !important;
        }}
        
        .prediction-box {{
            background-color: rgba(0, 255, 0, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            color: white !important;
        }}

        .big-table .stDataFrame {{
            width: 100vw !important;
            height: 80vh !important;
        }}

        .success-message {{
            background-color: rgba(0, 255, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #00FF00;
            margin-bottom: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set Background Image
set_background_image_local("Dark-bg.jpg")

# Load Model
car_model = load_model("carmodel2 (1).pkl")

# Load Dataset
df = pd.read_csv("car_price_ai.csv")

# Load Encoders
encoders = {
    "brand": load_model("encoder_brand.pkl"),
    "fuel_type": load_model("encoder_fuel_type.pkl"),
    "insurance_type": load_model("encoder_insurance_type.pkl"),
    "location": load_model("encoder_location.pkl"),
    "no_of_seats": load_model("encoder_no_of_seats.pkl"),
    "ownership": load_model("encoder_ownership.pkl"),
    "transmission": load_model("encoder_transmission.pkl"),
    "model": load_model("encoder_model.pkl")
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
    power_data = st.number_input("Power (BHP)", min_value=80.0, step=5.0)

# Additional Inputs
col4, col5 = st.columns(2)
with col4:
    cc = st.number_input("Engine CC", min_value=998.0, step=100.0)
with col5:
    mileage = st.number_input("Mileage (kmpl)", min_value=18.0, step=1.0)

# Normalize Skewed Data
km_driven = np.cbrt(km_driven_data)
power_transformed = stats.boxcox([power_data], lmbda=-0.5)[0]

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


# Car Model Detail Box
st.markdown("<h2 style='color: #FFA500;'>💬 Car Details Chatbot</h2>", unsafe_allow_html=True)

user_command = st.text_input("Enter command", placeholder="E.g., Give me detail info about Tata")


if st.button("🔍 Search Model"):
    if user_command.lower():
        selected_brand = user_command.split("about")[-1].strip()

        # Normalize case for better matching
        selected_brand = selected_brand.lower()  
        df["brand_lower"] = df["brand"].str.lower().str.strip()  

        # Check if the model exists
        model_info = df[df["brand_lower"] == selected_brand]
        if not model_info.empty:
            st.markdown("<div class='success-message'>✅ Model Found! Displaying Details Below:</div>", unsafe_allow_html=True)

            # Remove extra column before displaying
            model_info = model_info.drop(columns=["brand_lower"])
            
            # Maximize the result display
            st.dataframe(model_info, use_container_width=True)

        else:
            st.error("⚠️ Model details not found. Please check the spelling.")

