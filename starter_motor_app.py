# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:59:26 2024

@author: H244746
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression







st.cache(suppress_st_warning=True)













app_mode = st.sidebar.selectbox('Select Page',['Data Description','Starter Motor Fault Prediction']) #two pages



if app_mode == 'Data Description':
        
    data_path = 'C:\\Users\\H244746\\Desktop\\Feature Slider Example\\data_dictionary.xlsx'
    data_dict = pd.read_excel(data_path, sheet_name = 'Sheet1')
    
    
    
    st.image('apu_image.jpg')
    st.subheader('Column name descriptions')
    st.write(data_dict)
    
    
    
    
    


elif app_mode == 'Starter Motor Fault Prediction':
    # st.image('slider-short-3.jpg')
    
    # Get input data from app
    st.subheader('Please specify the time it takes to start the APU.')
    st.sidebar.header("Information about the APU :")
    start_time = st.sidebar.slider('Start Time',0,200,0,)





    # Create dictionary with input data
    data_from_app = {'start_time': start_time}
    
    
    
    # Load model pipeline
    model_path = 'C:\\Users\\H244746\\Desktop\\Feature Slider Example\\model\\starter_motor_classifier.sav'
    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)
      
        
    # Generate prediction for input data
    data_from_app = pd.DataFrame(data_from_app, index = [0])
    prediction = model.predict(data_from_app)
    prob = model.predict_proba(data_from_app)

    
    if st.button('Predict'):
        if prediction == 1 :
            st.error('According to the starter motor classification model, the starter motor is likely to experience a failure with a probability of {}%!'.format(round(prob[0][1] * 100),4))
            st.image('thumbs_down.jpg')
        
        elif prediction == 0:
            st.success('According to the starter motor classification model, the starter motor appears to be healthy with a probability of {}%.'.format(round(prob[0][0] * 100), 4))
            st.image('thumbs_up.jpg')