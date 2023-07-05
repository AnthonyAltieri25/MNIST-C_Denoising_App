import tensorflow as tf
from tensorflow import keras
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

buffer = io.BytesIO()
model = keras.models.load_model('my_autoencoder.keras')

st.title("SURP2023 MNIST-C Image Denoiser")

#Get the image from user
user_file = st.file_uploader("Enter a corrupted MNIST-C image", type=['jpg', 'png'], accept_multiple_files=False)

while(user_file is None):
    pass

st.write("Image received")

#Prepare image for autoencoder
image = Image.open(user_file)
arr = np.empty(shape=(1, 28, 28))
arr[0] = np.array(image)
arr = np.divide(arr.astype('float32'), 255)
arr = arr.reshape(len(arr), np.prod(arr.shape[1:]))

#Denoise image
st.write("DENOISING...")
arr = model.predict(arr)

#Reshape data
arr = np.reshape(arr, newshape=(1, 28, 28))
arr = np.multiply(arr, 255).astype('uint8')

#Create image fom data
image = Image.fromarray(obj=arr[0])
image.save(buffer, format='PNG')

#Give tuser button to display data
st.download_button(label='Download Denoised Image', data=buffer, file_name='Denoised.png')
