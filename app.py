import streamlit as st
import requests
from PIL import Image

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Plant Disease Predictor",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 AI Plant Disease Predictor")

st.write("Upload a leaf image and get instant prediction")

# -------------------------
# UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Convert image
    image = Image.open(uploaded_file)

    # Resize for UI (IMPORTANT FIX)
    image = image.resize((350, 350))  # 👈 controls height/width

    # -------------------------
    # LAYOUT (50/50 SCREEN SPLIT)
    # -------------------------
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Image Preview")
        st.image(image)

    with col2:
        st.subheader("🧠 Prediction Panel")

        st.write("Click below to predict disease")

        if st.button("🔍 Predict"):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )

                result = response.json()

                if response.status_code == 200:
                    st.success(f"🌱 Disease: {result['prediction']}")
                    st.info(f"📊 Confidence: {result['confidence']:.2f}")
                else:
                    st.error(result)

            except Exception as e:
                st.error(f"Error: {e}")