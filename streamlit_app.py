#import library needed
import pandas as pd
import streamlit as st 
import pickle
import numpy as np

# Load the best model
with open('iris_svm.pkl', 'rb') as f:
    model = pickle.load(f)

species = ['setosa', 'versicolor', 'virginica']

st.title("Iris Flower Classification")
st.write("This app correctly classifies iris flower among 3 possible species")

# Creating Sidebar for inputs
st.sidebar.title("Inputs")
sepal_length = st.sidebar.selectbox("sepal length (cm)", [0,1,2,3,4,5,6,7,8])
sepal_width = st.sidebar.slider("sepal width (cm)", 0, 8)
petal_length = st.sidebar.slider("petal length (cm)", 0, 8)
petal_width = st.sidebar.slider("petal width (cm)", 0, 8)

# Button to trigger prediction
if st.button("Predict"):
# Getting Prediction from model
    inp = np.array([sepal_length, sepal_width, petal_length, petal_width])
    inp = np.expand_dims(inp, axis=0)
    prediction = model.predict(inp)

# Show Results when the button is clicked
    result = species[prediction[0]]
    st.write("**This flower belongs to " + result + " class**")
    st.write(f"({prediction[0]})")