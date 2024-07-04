import streamlit as st
from keras_preprocessing.image import img_to_array, load_img
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained Keras model
try:
    model = load_model('cnn.h5')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

def preprocess_image(img):
    img = image.load_img(img, target_size=(64, 64))  # Load and resize image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values

st.set_page_config(page_title="Animal Classifier", layout="centered")

st.title("Classify animals")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    submit = st.button('Generate')

    if submit:
        try:
            img_array = preprocess_image(uploaded_file)
            prediction = model.predict(img_array)
            if prediction[0][0]>0.5:
                st.write("THE ANIMAL SHOWN IN THE IMAGE IS A DOG")
            else:
                st.write("THE ANIMAL SHOWN IN THE IMAGE IS A CAT")
        except Exception as e:
            st.error(f"Error predicting emotion: {str(e)}")
