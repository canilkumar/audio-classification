import streamlit as st
from keras.models import load_model
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(['No','Yes'])
classes = list(le.classes_)

customer_res_model = load_model('./Customer_Response_13.h5')

pricing_res_model = load_model('./Pricing_13.h5')

exclusions_res_model = load_model('./Insurance_Exclusion_13.h5')

claim_res_model = load_model('./Claim_Process_13.h5')

callClasification_res_model = load_model('./call_audio.h5')

exclusions_1_res_model = load_model('./Exclusions_1.h5')
exclusions_2_res_model = load_model('./Exclusions_2.h5')
exclusions_3_res_model = load_model('./Exclusions_3.h5')
exclusions_4_res_model = load_model('./Exclusions_4.h5')
exclusions_5_res_model = load_model('./Exclusions_5.h5')
exclusions_6_res_model = load_model('./Exclusions_6.h5')
exclusions_7_res_model = load_model('./Exclusions_7.h5')

daily_price_res_model = load_model('./Daily_Pricing.h5')
monthly_price_res_model = load_model('./Monthly_Pricing.h5')


def predict_callClasification_res(audio):
    prob=callClasification_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

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

def predict_exclusion_1(audio):
    prob=exclusions_1_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_2(audio):
    prob=exclusions_2_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_3(audio):
    prob=exclusions_3_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_4(audio):
    prob=exclusions_4_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_5(audio):
    prob=exclusions_5_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_6(audio):
    prob=exclusions_6_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_7(audio):
    prob=exclusions_7_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_daily_price(audio):
    prob=daily_price_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_monthly_price(audio):
    prob=monthly_price_res_model.predict(audio.reshape(1,40,1))
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
        
        call_res = predict_callClasification_res(featuesAll)
        st.title("Is a call : "+ call_res[0]+" Confidence: "+str(call_res[1]*100))

        if(call_res[0] =='Yes'):
         customer_res = predict_customer_res(featuesAll)
         st.title("customer response: "+ customer_res[0]+" Confidence: "+str(customer_res[1]*100))
         #st.title("customer response conf: "+ str(customer_res[1]))
         # " Confidence: "+customer_res[1]
         pricing_res = predict_pricing(featuesAll)
         st.title("pricing explained: "+ pricing_res[0]+" Confidence: "+str(pricing_res[1]*100))
        
         exclusions_res = predict_exclusion(featuesAll)
         st.title("exclusions explained: "+ exclusions_res[0]+" Confidence: "+str(exclusions_res[1]*100))
        
         claim_res = predict_claim(featuesAll)
         st.title("claim process explained: "+ claim_res[0]+" Confidence: "+str(claim_res[1]*100))

         exclusions_1_res = predict_exclusion_1(featuesAll)
         st.title("exclusions explained: "+ exclusions_1_res[0]+" Confidence: "+str(exclusions_1_res[1]*100))

         exclusions_2_res = predict_exclusion_2(featuesAll)
         st.title("exclusions explained: "+ exclusions_2_res[0]+" Confidence: "+str(exclusions_2_res[1]*100))
        
         exclusions_3_res = predict_exclusion_3(featuesAll)
         st.title("exclusions explained: "+ exclusions_3_res[0]+" Confidence: "+str(exclusions_3_res[1]*100))
         
         exclusions_4_res = predict_exclusion_4(featuesAll)
         st.title("exclusions explained: "+ exclusions_4_res[0]+" Confidence: "+str(exclusions_4_res[1]*100))
         
         exclusions_5_res = predict_exclusion_5(featuesAll)
         st.title("exclusions explained: "+ exclusions_5_res[0]+" Confidence: "+str(exclusions_5_res[1]*100))
        
         exclusions_6_res = predict_exclusion_6(featuesAll)
         st.title("exclusions explained: "+ exclusions_6_res[0]+" Confidence: "+str(exclusions_6_res[1]*100))
         
         exclusions_7_res = predict_exclusion_7(featuesAll)
         st.title("exclusions explained: "+ exclusions_7_res[0]+" Confidence: "+str(exclusions_7_res[1]*100))
        
        

        
        
       










