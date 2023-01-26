from numpy import argmax, array, expand_dims
from pandas import read_csv
import streamlit as st
from PIL import Image
from keras.models import load_model

model = load_model('C:/Users/Dev Gupta/Desktop/Jupyter/Projects (ML-DS)/Traffic Sign pred/traffic_classifier.h5')
labels = read_csv('C:/Users/Dev Gupta/Desktop/Jupyter/Projects (ML-DS)/Traffic Sign pred/labels.csv')

st.title('Traffic Sign Predictor')
# st.text(labels)

i_path = st.file_uploader('Upload an traffic sign Image',['.jpg','.png'],accept_multiple_files=False)
if i_path :
    st.image(i_path)
    img = Image.open(i_path)
    img = img.resize((30,30))
    img = expand_dims(img,axis=0)
    img = array(img)

    rough_pred = model.predict(img)
    pred = argmax(rough_pred,axis=1)
    class_ = labels.iloc[pred[0]].values[1]

    st.header('\t\t\t\t\t{}'.format(class_))