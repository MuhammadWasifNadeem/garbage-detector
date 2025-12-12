import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime
import base64

# -----------------------------
# ✅ Custom CSS for cleaner UI
# -----------------------------
st.markdown("""
    <style>
        .main { background-color: #F7F9FC; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
        }
        .uploadedImage {
            border-radius: 10px;
            border: 2px solid #ddd;
            padding: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# ✅ Sidebar Instructions
# -----------------------------
st.sidebar.title("📌 How to Use")
st.sidebar.write("""
1. Upload an image of waste  
2. Wait for the progress bar to finish  
3. View prediction + confidence  
4. Download your prediction report  
""")

st.sidebar.info("Supported formats: JPG, JPEG, PNG")

# -----------------------------
# ✅ Load TFLite GPU-optimized model
# -----------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="best_trash_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# -----------------------------
# ✅ Prediction Function (TFLite)
# -----------------------------
def predict_tflite(image):
    img = img_to_array(image) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    return output[0]

# -----------------------------
# ✅ Main App UI
# -----------------------------
st.title("♻️ Smart Trash Classification App")
st.write("Upload an image and let the AI classify the waste type.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------
# ✅ Prediction Logging Setup
# -----------------------------
LOG_FILE = "prediction_log.csv"

def log_prediction(filename, prediction, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[timestamp, filename, prediction, confidence]],
                         columns=["Timestamp", "Filename", "Prediction", "Confidence"])

    try:
        existing = pd.read_csv(LOG_FILE)
        updated = pd.concat([existing, entry], ignore_index=True)
    except:
        updated = entry

    updated.to_csv(LOG_FILE, index=False)

# -----------------------------
# ✅ Handle Uploaded Image
# -----------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Progress bar
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    # Load and preprocess
    img = load_img(uploaded_file, target_size=(180, 180))

    # Predict
    preds = predict_tflite(img)
    pred_class = class_labels[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)

    # Log prediction
    log_prediction(uploaded_file.name, pred_class, confidence)

    # Display result
    st.subheader(f"✅ Prediction: **{pred_class.capitalize()}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # -----------------------------
    # ✅ Confidence Bar Chart
    # -----------------------------
    chart_data = pd.DataFrame({
        "Class": class_labels,
        "Confidence": preds * 100
    })

    chart = alt.Chart(chart_data).mark_bar().encode(
        x="Class",
        y="Confidence",
        color="Class"
    )

    st.altair_chart(chart, use_container_width=True)

    # -----------------------------
    # ✅ Downloadable Prediction Report
    # -----------------------------
    report = f"""
    Trash Classification Report
    ---------------------------
    File: {uploaded_file.name}
    Prediction: {pred_class}
    Confidence: {confidence:.2f}%
    Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """

    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="prediction_report.txt">📥 Download Prediction Report</a>'
    st.markdown(href, unsafe_allow_html=True)
