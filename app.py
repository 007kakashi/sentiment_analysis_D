# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# from keras.utils.data_utils import pad_sequences
import regex as re
from sklearn.feature_extraction.text import CountVectorizer
from numpy import loadtxt
from keras.models import load_model

# load model
# classifier = load_model('model.h5')

open 

filenametoken = 'model.pkl'
classifier = pickle.load(open(filenametoken, 'rb'))

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# def clean(text):
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'http\S+', '', text)
#     text = text.lower()
#     text = re.sub(r'\d+', '', text)
#     text = ''.join([i for i in text if i.isalpha() or i.isspace()])
#     tokens = nltk.word_tokenize(text)
#     lemma= WordNetLemmatizer()
#     tokens = [lemma.lemmatize(i) for i in tokens if not i in stopwords.words('english')]
#     text = ' '.join(tokens)
#     print(text)

#     return text

with open('cleaner.pkl','rb') as file:
    cleaner=pickle.load(file)

def predict(text):
    Class = classifier.predict(text,batch_size=1,verbose = 2)
    print(Class)
    y_predicted = Class.flatten()
    print(y_predicted)
    if y_predicted==0:
        return "Regular"
    elif y_predicted==1:
        return "Sarcasm"
    elif y_predicted==2:
        return "Irony"
    elif y_predicted==3:     
        return "Figurative"
    

def main():

  add_selectbox = st.selectbox("How would you like to predict?",("Sentence", "Batch File"))

  st.title("Twitter Sentiment Analysis")

  if add_selectbox == 'Sentence':
      Sentence = st.text_input("Enter Text",key ="account_lenght")  
     
    
      result=""
      if st.button("Predict"):
          cleaning=cleaner(Sentence)

          result=predict(cleaning)


      st.success('{}'.format(result))
      
if __name__=='__main__':
    main()




    

