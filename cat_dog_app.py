import joblib
import streamlit as st
import numpy as np

model = joblib.load("cat_dog_classification_model.pkl")

st.title("Welcome to my Cat Dog Classifier App")

col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths to balance alignment
with col2:
    st.image("catdog.jpg", use_column_width="auto")

st.write("Enter the features to classify whether it's a Cat or a Dog.")

# Input features
height = st.number_input("Height (in cm):", min_value=10.0, max_value=100.0, value=30.0)
weight = st.number_input("Weight (in kg):", min_value=1.0, max_value=50.0, value=5.0)
sound = st.radio("Sound:", options=["Meow", "Bark"])

# Convert sound to numeric value
sound_value = 0 if sound == "Meow" else 1

# Predict button
if st.button("Predict"):
    # Prepare the feature array
    features = np.array([[height, weight, sound_value]])
    prediction = model.predict(features)
    if sound_value == 0:
        result = "Cat"
    else:
        result = "Dog" if (prediction[0] == 1 or sound_value == 1) else "Cat"
    st.write(f"The predicted animal is: **{result}**")