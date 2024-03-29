import streamlit as st
import tensorflow as tf
import os
from keras.models import load_model
from PIL import Image
import numpy as np
import subprocess
if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://github.com/PritishMaske/Disease-detection-in-orange-fruit-plantation/blob/main/web-app_1/model/training_of_leaves.h5"'], shell=True)
from util import classify, set_background


set_background('./bg.png')

# set title
st.markdown('<h1 style="color: white;">Disease Detection in Orange Fruit</h1>', unsafe_allow_html=True)

# set header
st.markdown('<h2 style="color: white;">Please upload image</h2>', unsafe_allow_html=True)

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = tf.keras.models.load_model('model.h5', compile=False)

# load class names
with open('./web-app_1/model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB').resize((256, 256))
    st.image(image, use_column_width=True)

    # classify image
    predicted_class, confidence = classify(image, model, class_names)

    # write classification
    st.write('<h2 style="color: white;">Predicted Disease: {}</h2>'.format(predicted_class), unsafe_allow_html=True)
    st.write('<h3 style="color: white;">Confidence score: {}%</h3>'.format(int(confidence * 1000) / 10), unsafe_allow_html=True)
    
