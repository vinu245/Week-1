import streamlit as st
import pickle
import numpy as np

# Load the MinMaxScaler and the model
with open("minmaxscaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Crop dictionary for mapping model output to crop names
crop_dict = {
    
    1: "Wheat",
    2: "Maize",
    3: "Jute",
    4: "Cotton",
    5: "Coconut",
    6: "Papaya",
    7: "Orange",
    8: "Apple",
    9: "Muskmelon", 
    10: "Watermelon", 
    11: "Grapes", 
    12: "Mango",
    13: "Banana",
    14: "Pomegranate", 
    15: "Lentil", 
    16: "Blackgram", 
    17: "Mungbean", 
    18: "Mothbeans",
    19: "Pigeonpeas", 
    20: "Kidneybeans", 
    21: "Chickpea", 
    22: "Coffee"
    
    
}

# Function to make predictions
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Input as a numpy array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the input
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    # Map the numerical prediction to the crop name
    crop_prediction = crop_dict.get(int(prediction[0]), "Unknown crop")
    
    return crop_prediction

# Streamlit interface
st.title("Crop Prediction Recommendation System")

# Create input fields
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, step=0.1)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, step=0.1)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, step=0.1)
temperature = st.number_input("Temperature", min_value=10.0, max_value=60.0, step=0.1)
humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall", min_value=0.0, max_value=500.0, step=0.1)

# Button to trigger prediction
if st.button("Predict Crop"):
    result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    st.success(f'The predicted crop is: {result}')