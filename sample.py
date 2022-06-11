import streamlit as st
from keras.models import load_model
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(['No','Yes'])
classes = list(le.classes_)

customer_res_model = load_model('./Cr_2.h5')

pricing_res_model = load_model('./Pricing_13.h5')

exclusions_res_model = load_model('./Insurance_Exclusion_13.h5')

claim_res_model = load_model('./Cp_3.h5')

callClasification_res_model = load_model('./call_audio.h5')

exclusions_1_res_model = load_model('./ex_1.h5')
exclusions_2_res_model = load_model('./ex_2.h5')
exclusions_3_res_model = load_model('./ex_3.h5')
exclusions_4_res_model = load_model('./ex_4.h5')
exclusions_5_res_model = load_model('./ex_5.h5')
exclusions_6_res_model = load_model('./ex_6.h5')
exclusions_7_res_model = load_model('./ex_7.h5')
exclusions_8_res_model = load_model('./ex_8.h5')
exclusions_9_res_model = load_model('./ex_9.h5')
exclusions_10_res_model = load_model('./ex_10.h5')
exclusions_11_res_model = load_model('./ex_11.h5')
exclusions_12_res_model = load_model('./ex_12.h5')


daily_price_res_model = load_model('./Daily_Pricing_3.h5')
monthly_price_res_model = load_model('./Monthly_Pricing_3.h5')


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

def predict_exclusion_8(audio):
    prob=exclusions_8_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_9(audio):
    prob=exclusions_9_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_10(audio):
    prob=exclusions_10_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_11(audio):
    prob=exclusions_11_res_model.predict(audio.reshape(1,40,1))
    index=np.argmax(prob[0])
    return (classes[index],np.amax(prob[0]))

def predict_exclusion_12(audio):
    prob=exclusions_12_res_model.predict(audio.reshape(1,40,1))
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

def boldTag(status):
    if (status=='Yes'):
        return "<b style=color:green;>"
    return "<b style=color:red;>"
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

        
        # st.title("1.By consuming alcohol if you facing any dieses and death then policy will not claim")
        # st.title("2.If you not following doctors advice then policy will not claimed")
        # st.title("3.If you attempt suicide or any other thing which is affected you then policy will not claimed")
        # st.title("4.If Natural Calamity( Disaster) Happens that time if Government Declair Emergency then Policy is not Eligible")
        # st.title("5.If you infect and died during country war when country has declared emergency then policy will not claimed")
        # st.title("6.If you face any infection during cosmetic treatment and the policy is not more than 9 month then you will not claimed")
        # st.title("7.Before 3 months you have cancer and blindness then apply for claim then you not eligible for claim process")

        
        call_res = predict_callClasification_res(featuesAll)
        st.markdown("Is it a call: "+boldTag(call_res[0]) + call_res[0]+"</b> Confidence: <b>"+str(call_res[1]*100)+"</b>", unsafe_allow_html=True)

        # st.write("Is a call : "+ call_res[0]+" Confidence: "+str(call_res[1]*100))

        
        if(call_res[0] =='Yes'):
         customer_res = predict_customer_res(featuesAll)
         st.markdown("customer response: "+boldTag(customer_res[0])+ customer_res[0]+"</b> Confidence: <b>"+str(customer_res[1]*100)+"</b>", unsafe_allow_html=True)
         #st.write("customer response: "+ customer_res[0]+" Confidence: "+str(customer_res[1]*100))
         #st.title("customer response conf: "+ str(customer_res[1]))
         # " Confidence: "+customer_res[1]
        #  pricing_res = predict_pricing(featuesAll)
        #  st.title("pricing explained: "+ pricing_res[0]+" Confidence: "+str(pricing_res[1]*100))
        
        #  exclusions_res = predict_exclusion(featuesAll)
        #  st.title("exclusions explained: "+ exclusions_res[0]+" Confidence: "+str(exclusions_res[1]*100))
        
         claim_res = predict_claim(featuesAll)
         st.markdown("claim process explained: "+boldTag(claim_res[0])+ claim_res[0]+"</b> Confidence: <b>"+str(claim_res[1]*100)+"</b>", unsafe_allow_html=True)

         #st.write("claim process explained: "+ claim_res[0]+" Confidence: "+str(claim_res[1]*100))

         st.title("Exclusion Points explanation")

         exclusions_1_res = predict_exclusion_1(featuesAll)
         st.markdown("1.Khudkushi explained: "+boldTag(exclusions_1_res[0])+ exclusions_1_res[0]+"</b> Confidence: <b>"+str(exclusions_1_res[1]*100)+"</b>", unsafe_allow_html=True)

         exclusions_2_res = predict_exclusion_2(featuesAll)
         st.markdown("2.Halat Jang explained: "+boldTag(exclusions_2_res[0])+ exclusions_2_res[0]+"</b> Confidence: <b>"+str(exclusions_2_res[1]*100)+"</b>", unsafe_allow_html=True)
        
         exclusions_3_res = predict_exclusion_3(featuesAll)
         st.markdown("3.Agwa Baray Tawan explained: "+boldTag(exclusions_3_res[0])+ exclusions_3_res[0]+"</b> Confidence: <b>"+str(exclusions_3_res[1]*100)+"</b>", unsafe_allow_html=True)
         
         exclusions_4_res = predict_exclusion_4(featuesAll)
         st.markdown("4.Manshiyat ya kisi b Nasha Awar cheez k Istemal se Nuqsan explained: "+boldTag(exclusions_4_res[0])+ exclusions_4_res[0]+"</b> Confidence: <b>"+str(exclusions_4_res[1]*100)+"</b>", unsafe_allow_html=True)
         
         exclusions_5_res = predict_exclusion_5(featuesAll)
         st.markdown("5.Qudarti Afatya Dehshat Gardana Karwai me bare pemane pr hone walay Nuqsan me Haqumat Emergency ka Alan kr de ya explained: "+boldTag(exclusions_5_res[0])+ exclusions_5_res[0]+"</b> Confidence: <b>"+str(exclusions_5_res[1]*100)+"</b>", unsafe_allow_html=True)
        
         exclusions_6_res = predict_exclusion_6(featuesAll)
         st.markdown("6.Civil Nafarmani ki surat mea policy claim nahi hoge explained: "+boldTag(exclusions_6_res[0])+ exclusions_6_res[0]+"</b> Confidence: <b>"+str(exclusions_6_res[1]*100)+"</b>", unsafe_allow_html=True)
         
         exclusions_7_res = predict_exclusion_7(featuesAll)
         st.markdown("7.Khud ko jaan bhooj kar pohanchaye Janay walay nuqsan explained: "+boldTag(exclusions_7_res[0])
         + exclusions_7_res[0]+"</b> Confidence: <b>"+str(exclusions_7_res[1]*100)+"</b>", unsafe_allow_html=True)
         
         exclusions_8_res = predict_exclusion_8(featuesAll)
         st.markdown("8.khudkhushi ki koshish explained: "+boldTag(exclusions_8_res[0])+ exclusions_8_res[0]+"</b> Confidence: <b>"+str(exclusions_8_res[1]*100)+"</b>", unsafe_allow_html=True)
         
         exclusions_9_res = predict_exclusion_9(featuesAll)
         st.markdown("9.doctor k mashawary se laparwahi ki surat mein honay wala nuqsan explained: "+boldTag(exclusions_9_res[0])+ exclusions_9_res[0]+"</b> Confidence: <b>"+str(exclusions_9_res[1]*100)+"</b>", unsafe_allow_html=True)
         
         exclusions_10_res = predict_exclusion_10(featuesAll)
         st.markdown("10.cosmentic surgery jo k ghair zaroori ho jaisa k chehray ki khoobsurti barhanay jaisay treatment explained: "+boldTag(exclusions_10_res[0])+ exclusions_10_res[0]+"</b> Confidence: <b>"+str(exclusions_10_res[1]*100)+"</b>", unsafe_allow_html=True)
         
         exclusions_11_res = predict_exclusion_11(featuesAll)
         st.markdown("11.Doran e hamal honay wali koi bhi paicheedgi jab k sarif ko service liye huay 9 maheenay ka arsa na hua ho explained: "+boldTag(exclusions_11_res[0])+ exclusions_11_res[0]+"</b> Confidence: <b>"+str(exclusions_11_res[1]*100)+"</b>", unsafe_allow_html=True)
         
         exclusions_12_res = predict_exclusion_12(featuesAll)
         st.markdown("12.Koi bhi baari tibi bemari jis ki tashkhees coverage shuru honay se 3 maah pehlay hoi ho, policy claim nahi ho gi explained: "+boldTag(exclusions_12_res[0])+ exclusions_12_res[0]+"</b> Confidence: <b>"+str(exclusions_12_res[1]*100)+"</b>", unsafe_allow_html=True)

         daily_price_res = predict_daily_price(featuesAll)
         st.markdown("daily price explained: "+boldTag(daily_price_res[0])+ daily_price_res[0]+"</b> Confidence: <b>"+str(daily_price_res[1]*100)+"</b>", unsafe_allow_html=True)
         
         monthly_price_res = predict_monthly_price(featuesAll)
         st.markdown("monthly price explained: "+boldTag(monthly_price_res[0])+ monthly_price_res[0]+"</b> Confidence: <b>"+str(monthly_price_res[1]*100)+"</b>", unsafe_allow_html=True)
        
        

        
        
       










