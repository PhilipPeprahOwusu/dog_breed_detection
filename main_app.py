import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading the model
model = load_model('dog_breed.h5')

# Name of dog classes
dog_names = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

# Setting the title of the app
st.title("Dog Breed Prediction")
st.markdown("Upload an image of a dog")

# Uploading image of a dog
dog_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

submit = st.button('Predict')

# On predict button click
if submit:
    if dog_image is None:
        st.write("Please upload an image.")
    else:
        try:
            # Converting file to OpenCV image
            file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Displaying the image
            st.image(opencv_image, channels="BGR")

            # Resizing the image while preserving aspect ratio
            resized_image = cv2.resize(opencv_image, (224, 224))
            opencv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            # Convert image to 4 dimensions
            input_image = np.expand_dims(opencv_image, axis=0)

            # Make prediction
            predictions = model.predict(input_image)
            predicted_class_index = np.argmax(predictions)
            predicted_breed = dog_names[predicted_class_index]

            st.title(f"The predicted dog breed is {predicted_breed}")
        except Exception as e:
            st.write("Error processing the image:", str(e))