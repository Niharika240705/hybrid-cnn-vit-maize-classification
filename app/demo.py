import os

import requests
import streamlit as st


API_URL = os.getenv("MAIZE_API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="Maize Seed Classifier", page_icon="🌽")
st.title("Maize Seed Classifier")
st.caption("Upload a seed image to classify its quality.")

uploaded_file = st.file_uploader("Seed image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Selected image", use_container_width=True)
    if st.button("Predict", type="primary"):
        try:
            response = requests.post(
                API_URL,
                files={"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            st.metric("Prediction", result["class"], f"{result['confidence']:.1%} confidence")
        except requests.RequestException as error:
            st.error(f"Prediction service unavailable: {error}")