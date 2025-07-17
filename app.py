import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('model1.h5')

st.title("Fake Drug Detection App")
st.write("Upload an image of the drug to detect if it's real or fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    img = img.resize((150,150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("⚠️ Likely a **FAKE** drug.")
    else:
        st.success("✅ Likely a **REAL** drug.")
