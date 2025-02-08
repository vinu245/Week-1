import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load models and scalers
def load_models_and_scalers():
    crop_model = joblib.load('Random.pkl')  # Crop recommendation model
    fertilizer_model = joblib.load('model_ftz.pkl')  # Fertilizer recommendation model
    crop_scaler = joblib.load('std_scaler.pkl')  # Scaler for Crop Recommendation
    fertilizer_scaler = joblib.load('scaler_ftz.pkl')  # Scaler for Fertilizer Recommendation
    return crop_model, fertilizer_model, crop_scaler, fertilizer_scaler

# Crop Mapping (Numbering crops 1 to 22)
CROP_MAPPING = {
    "Rice": 1, "Maize": 2, "Chickpea": 3, "Kidney Beans": 4, "Pigeon Peas": 5,
    "Moth Beans": 6, "Mung Bean": 7, "Black Gram": 8, "Lentil": 9, "Pomegranate": 10,
    "Banana": 11, "Mango": 12, "Grapes": 13, "Watermelon": 14, "Muskmelon": 15,
    "Apple": 16, "Orange": 17, "Papaya": 18, "Coconut": 19, "Cotton": 20,
    "Jute": 21, "Coffee": 22
}

# Home Page
def home_page():
    st.title("Welcome to the Agricultural Assistance System")
    st.subheader("This system provides crop and fertilizer recommendations.")
    st.write("""
        - *Crop Recommendation*: Enter environmental and soil parameters to get crop recommendations.
        - *Fertilizer Recommendation*: Enter crop and soil details to get the appropriate fertilizer recommendations.
    """)
    st.write("Navigate through the options above to get started!")

# Crop Recommendation Page
def crop_recommendation_page(crop_model, crop_scaler):
    st.title("Crop Recommendation System")
    st.subheader("Enter the required parameters to get crop recommendations")

    # Input fields for crop recommendation
    inputs = {
        "temperature": st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1),
        "humidity": st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1),
        "pH": st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1),
        "rainfall": st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0, step=1.0),
        "N": st.number_input("Nitrogen (N) content in soil", min_value=0.0, max_value=150.0, value=20.0, step=1.0),
        "P": st.number_input("Phosphorous (P) content in soil", min_value=0.0, max_value=150.0, value=30.0, step=1.0),
        "K": st.number_input("Potassium (K) content in soil", min_value=0.0, max_value=200.0, value=20.0, step=1.0)
    }

    if st.button("Recommend Crop"):
        try:
            # Prepare input data for crop recommendation
            input_data = pd.DataFrame([[inputs["temperature"], inputs["humidity"], inputs["pH"], inputs["rainfall"]]], 
                                      columns=['temperature', 'humidity', 'ph', 'rainfall'])
            
            # Scale using crop scaler
            scaled_features = crop_scaler.transform(input_data)
            
            # Combine scaled features with N, P, K values
            final_data = np.hstack([scaled_features, [[inputs["N"], inputs["P"], inputs["K"]]]])

            # Predict the crop
            recommendation = crop_model.predict(final_data)

            # Display recommendation
            st.success(f"Recommended Crop: {recommendation[0].upper()}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Show sample data
    st.subheader("Sample Crop Recommendations")
    if 'crop_samples' not in st.session_state or st.button("Refresh Crop Samples"):
        st.session_state.crop_samples = pd.read_csv('Crop_recommendation.csv')
    st.write(st.session_state.crop_samples.sample(5))

# Fertilizer Recommendation Page
def fertilizer_recommendation_page(fertilizer_model, fertilizer_scaler):
    st.title("Fertilizer Recommendation System")
    st.subheader("Enter crop and soil details to get fertilizer recommendations")

    # User inputs
    crop = st.selectbox("Select Crop", list(CROP_MAPPING.keys()))
    inputs = {
        "temperature": st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1),
        "humidity": st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1),
        "pH": st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1),
        "rainfall": st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
    }

    # Map the selected crop to its corresponding label
    crop_label = CROP_MAPPING[crop]

    if st.button("Recommend Fertilizer"):
        try:
            # Prepare input data for fertilizer recommendation
            input_data = np.array([[inputs["temperature"], inputs["humidity"], inputs["pH"], inputs["rainfall"], crop_label]])

            # Scale using fertilizer scaler
            scaled_data = fertilizer_scaler.transform(input_data)

            # Predict the fertilizer
            recommendation = fertilizer_model.predict(scaled_data)

            # Assuming the recommendation is a list of N, P, K values
            n, p, k = recommendation[0]
            st.success(f"Recommended Fertilizer for {crop}:")
            st.json({
                "Nitrogen (N)": f"{n} Kg/hactare",
                "Phosphorous (P)": f"{p} Kg/hactare",
                "Potassium (K)": f"{k} Kg/hactare"
            })
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Show sample data
    st.subheader("Sample Fertilizer Recommendations")
    if 'fertilizer_samples' not in st.session_state or st.button("Refresh Fertilizer Samples"):
        st.session_state.fertilizer_samples = pd.read_csv('Fertilizer Prediction.csv')
    st.write(st.session_state.fertilizer_samples.sample(5))

# Main function to run the app
def main():
    # Load models and scalers
    crop_model, fertilizer_model, crop_scaler, fertilizer_scaler = load_models_and_scalers()

    # Create tabs for navigation
    tabs = st.selectbox("Select a window", ["Home", "Crop Recommendation", "Fertilizer Recommendation"])

    if tabs == "Home":
        home_page()
    elif tabs == "Crop Recommendation":
        crop_recommendation_page(crop_model, crop_scaler)
    elif tabs == "Fertilizer Recommendation":
        fertilizer_recommendation_page(fertilizer_model, fertilizer_scaler)

# Run the app
if __name__ == "_main_":
    main()