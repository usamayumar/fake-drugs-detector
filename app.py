import streamlit as st
import numpy as np
from PIL import Image
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

st.title("Fake Drug Detection App")
st.write("Upload the trained model (`model1.h5`) and an image of the drug to detect if it's real or fake.")

# Upload model
model_file = st.file_uploader("Upload your model (.h5)", type=["h5"])

if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(model_file.read())
        model = load_model(tmp_file.name)

    # Upload image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        img = img.resize((150,150))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)

        if prediction[0][0] > 0.5:
            st.error("⚠️ Likely a **FAKE** drug.")
        else:
            st.success("✅ Likely a **REAL** drug.")
