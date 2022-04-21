import streamlit as st
from keras.models import load_model
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(['No','Yes'])
classes = list(le.classes_)

customer_res_model = load_model('./Customer_Response_1.h5')

pricing_res_model = load_model('./Pricing_1.h5')

exclusions_res_model = load_model('./Insurance_Exclusion_1.h5')

claim_res_model = load_model('./Claim_Process_1.h5')


def predict_customer_res(audio):
    prob=customer_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))
    
def predict_pricing(audio):
    prob=pricing_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion(audio):
    prob=exclusions_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_claim(audio):
    prob=claim_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))
    
def extract_feature(file_name):
      X, sample_rate = librosa.load(file_name)
      feature = np.array([])
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
      features = np.hstack((feature, mfccs))
      return features
      
def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('audio',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0      
from helper import *

#importing all the helper fxn from helper.py which we will create later

import streamlit as st

import os

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme(style="darkgrid")

sns.set()

from PIL import Image

st.title('Audio Classifier')

uploaded_file = st.file_uploader("Upload audio")

# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 

        # display the image
        test_sample1 = "audio/"+uploaded_file.name
        data, sampling_rate = librosa.load(test_sample1)
        
        featuesAll=extract_feature(test_sample1)
        
        customer_res = predict_customer_res(featuesAll)
        st.title("customer response: "+ customer_res[0]+" Confidence: "+customer_res[1])
         
        pricing_res = predict_pricing(featuesAll)
        st.title("pricing explained: "+ pricing_res[0]+" Confidence: "+pricing_res[1])
        
        exclusions_res = predict_exclusion(featuesAll)
        st.title("exclusions explained: "+ exclusions_res[0]+" Confidence "+exclusions_res[1])
        
        claim_res = predict_claim(featuesAll)
        st.title("claim process explained: "+ claim_res[0]+" Confidence "+claim_res[1])
        











