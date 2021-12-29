import streamlit as st
st.header("Common Papaya Disease Classifier")
st.text ("Provide a image of common papaya disease for image classification")

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import h5py
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import preprocessing
from keras.models import load_model
from keras import models

def main():
  file_uploaded= st.file_uploader("Choose the file", type=["jpg","jpeg" ,"png"])
  if file_uploaded is not None:
    image=Image.open(file_uploaded)
    figure=plt.figure()
    plt.imshow(image)
    plt.axis('off')
    result=predict_class(image)
    st.write(result)
    st.pyplot(figure)
  else:
    st.text("Please upload an image file")
    
def predict_class(image):
  model=models.load_model(r"C:\Users\Rumesha\Desktop\ru\papaya\papaya\cnn.h5")
  shape=((256,256,3))
  tf.keras.Sequential([hub.KerasLayer(model,input_shape=shape)])
  test_image=image.resize((256,256))
  test_image=preprocessing.image.img_to_array(test_image)
  test_image=test_image/255.0
  test_image=np.expand_dims(test_image,axis=0)
  class_names=['Anthracnose', 'Black spot', 'Healthy', 'Phytopthora', 'Powdery mildew', 'Ring spot']
  
  predictions = model.predict(test_image)
  scores=tf.nn.softmax(predictions[0])
  scores= scores.numpy()
  image_class=class_names[np.argmax(scores)]
  results="This image most likely belongs to {}with a {:.2f} percent confidence.",format(image_class), 100 * np.max(scores)

  return results

if __name__ == "__main__":
    main() 
