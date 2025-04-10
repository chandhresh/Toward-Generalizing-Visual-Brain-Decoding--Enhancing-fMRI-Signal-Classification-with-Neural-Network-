import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Single Person fMRI Analyzer", layout="centered")

# Safe model loading
@st.cache_resource
def load_trained_model():
    try:
        return load_model("fmri_brain_model.keras", compile=False)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_trained_model()

# Region label mapping
region_labels = {
    0: "Hippocampus (Alzheimer's Risk)",
    1: "Prefrontal Cortex (Normal Cognition)",
    2: "Motor Cortex",
    3: "Visual Cortex",
    4: "Cerebellum"
}

# Background image setup
def set_background_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_from_url("https://images.unsplash.com/photo-1581090700227-1e8e02b1e9d9?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80")

# Title
st.markdown("<h1 style='color: white;'>🧠 Single Person fMRI Analyzer</h1>", unsafe_allow_html=True)

# Upload fMRI file
uploaded_file = st.file_uploader("📁 Upload .npz fMRI File", type=["npz"])

if uploaded_file and model:
    try:
        data = np.load(uploaded_file)
        if "bold" not in data:
            st.error("❌ The uploaded .npz file must contain a 'bold' array.")
        else:
            bold_data = data["bold"]
            st.success("✅ fMRI file loaded successfully.")

            # User input for BOLD signals
            user_input = st.text_area("✍️ Enter BOLD Signals (comma-separated):")

            if st.button("🔍 Predict"):
                try:
                    signals = [float(x.strip()) for x in user_input.strip().split(",")]
                    signal_array = np.array(signals).reshape(-1, 1)

                    predictions = model.predict(signal_array)
                    predicted_labels = np.argmax(predictions, axis=1)
                    confidence_scores = [f"{np.max(p) * 100:.2f}%" for p in predictions]
                    probabilities = [p.tolist() for p in predictions]

                    normal_range = (-0.1, 0.1)

                    for i, (sig, label, conf, prob) in enumerate(zip(signals, predicted_labels, confidence_scores, probabilities)):
                        region = region_labels.get(label, f"Region {label}")
                        anomaly = "Yes" if sig < normal_range[0] or sig > normal_range[1] else "No"
                        condition = "Potential Risk" if anomaly == "Yes" and label == 0 else "Normal"

                        st.markdown(f"""
                        <div style='background-color: rgba(0,0,0,0.6); padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                        <b>Signal {i+1}:</b> {sig}<br>
                        - Predicted Region: <span style='color: #00ccff;'><b>{region}</b></span><br>
                        - Confidence: <b>{conf}</b><br>
                        - Probabilities: {prob}<br>
                        - Anomaly: <b>{anomaly}</b><br>
                        - Associated Condition: <b>{condition}</b>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"⚠️ Prediction error: {e}")
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
