# app.py
import streamlit as st
from pathlib import Path
import numpy as np
import joblib
import os

st.set_page_config(page_title="Traffic Detection App", layout="centered")

st.title("Traffic Detection — Simple Demo")
st.write("Upload an image or enter numeric features to get a prediction from the saved model.")

# Path to model file inside repo
MODEL_PATH = Path("model") / "traffic_model.pkl"

@st.cache_resource
def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

st.sidebar.header("Input options")
input_mode = st.sidebar.radio("Choose input mode", ["Numeric features", "Upload image (placeholder)"])

if input_mode == "Numeric features":
    st.header("Numeric input")
    col1, col2 = st.columns(2)
    with col1:
        speed = st.number_input("Vehicle speed (km/h)", value=40.0, min_value=0.0, step=0.5)
        weight = st.number_input("Vehicle weight (kg)", value=1500.0, min_value=0.0, step=10.0)
    with col2:
        time_of_day = st.selectbox("Time of day (informational)", ["morning", "afternoon", "evening", "night"])

    if st.button("Predict"):
        if model is None:
            st.error("Model file not found. Make sure `model/traffic_model.pkl` exists in the repo.")
        else:
            try:
                # Adjust features to match your trained model
                X = np.array([[speed, weight]])
                pred = model.predict(X)
                st.success(f"Prediction: {pred[0]}")
            except Exception as e:
                st.error(f"Model prediction failed: {e}")

else:
    st.header("Image upload (placeholder)")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file)
        st.info("Image inference not implemented in this template. Replace with your own preprocessing & inference code.")

st.markdown("---")
st.caption("Template app — adapt model loading & preprocessing to your use-case.")

# For debugging: show working dir and model existence when run with debug checkbox
if st.sidebar.checkbox("Show debug info"):
    st.write("Working directory:", os.getcwd())
    st.write("Model path:", str(MODEL_PATH))
    st.write("Model exists:", MODEL_PATH.exists())
