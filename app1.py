# app.py
import streamlit as st
st.set_page_config(page_title="Image Classifier", layout="centered")

import numpy as np
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt

# ---------------------------
# Load TFLite Model
# ---------------------------
st.title("🔍Image Classification - Real vs Recaptured")
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="mobile_TEST.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# ---------------------------
# Get Input & Output Details
# ---------------------------
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
DATA_FORMAT = "NCHW" if input_shape[1] == 3 else "NHWC"
IMAGE_SIZE = (input_shape[-2], input_shape[-1]) if DATA_FORMAT == "NCHW" else (input_shape[1], input_shape[2])

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    
    arr = np.expand_dims(arr, axis=0)  # Add batch dim
    
    if DATA_FORMAT == "NCHW":
        arr = np.transpose(arr, (0, 3, 1, 2))
    return arr

def predict_tflite(img_arr: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_details[0]["index"], img_arr.astype(input_details[0]["dtype"]))
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]["index"])
    return logits


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# ---------------------------
# File Upload + UI Layout
# ---------------------------
st.sidebar.header("📁 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

# ---------------------------
# Main Panel
# ---------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    img_arr = preprocess_image(image)
    logits = predict_tflite(img_arr)
    probs = softmax(logits)[0]
    
    class_names = ["Real", "Recaptured"]
    pred_class_index = int(np.argmax(probs))
    pred_class = class_names[pred_class_index]

    # 🟢 Predicted class highlighted
    st.markdown(f"<h2 style='color:green;'>✅ Predicted Class: <strong>{pred_class}</strong></h2>", unsafe_allow_html=True)

    # 📊 Progress bars for each class
    st.write("### 🔢 Prediction Confidence")
    for i, prob in enumerate(probs):
        st.progress(float(prob), text=f"{class_names[i]}: {prob*100:.2f}%")

    # 📈 Bar chart visualization
    st.write("### 📉 Class Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar(class_names, probs * 100, color=['#6c5ce7', '#00cec9'])
    ax.set_ylabel('Probability (%)')
    ax.set_ylim([0, 100])
    ax.set_title("Class Probabilities")
    st.pyplot(fig)
else:
    st.info("👈 Please upload an image from the sidebar to get started.")
# col1, col2 = st.columns([1, 3])

# with col1:
#     uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# with col2:
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

#         img_arr = preprocess_image(image)
#         logits = predict_tflite(img_arr)
#         probs = softmax(logits)[0]
        
#         class_names = ["Real", "Recaptured"]
#         pred_class_index = int(np.argmax(probs))
#         pred_class = class_names[pred_class_index]

#         # 🟢 Predicted class highlighted
#         st.markdown(f"<h2 style='color:green;'>✅ Predicted Class: <strong>{pred_class}</strong></h2>", unsafe_allow_html=True)

#         # 📊 Progress bars for each class
#         st.write("### 🔢 Prediction Confidence")
#         for i, prob in enumerate(probs):
#             st.progress(float(prob), text=f"{class_names[i]}: {prob*100:.2f}%")

#         # 📈 Bar chart visualization
#         st.write("### 📉 Class Probability Distribution")
#         fig, ax = plt.subplots()
#         ax.bar(class_names, probs * 100, color=['#6c5ce7', '#00cec9'])
#         ax.set_ylabel('Probability (%)')
#         ax.set_ylim([0, 100])
#         ax.set_title("Class Probabilities")
#         st.pyplot(fig)
