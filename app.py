import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
@st.cache_resource
def load_trained_model():
    model = load_model('pizza_steak_classifier.h5')
    return model

model = load_trained_model()

# Streamlit app title and description
st.title("🍕🥩 Pizza vs Steak Classifier")
st.write("Upload an image of pizza or steak, and the model will predict which it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    # Show result
    if confidence < 0.5:
        st.success(f"🍕 Pizza (Confidence: {100*(1-confidence):.1f}%)")
    else:
        st.success(f"🥩 Steak (Confidence: {100*confidence:.1f}%)")
