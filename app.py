# Import necessary libraries
import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

filenametoken = 'model.pkl'
classifier = pickle.load(open(filenametoken, 'rb'))


def clean(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = ''.join([i for i in text if i.isalpha() or i.isspace()])
    tokens = nltk.word_tokenize(text)
    lemma= WordNetLemmatizer()
    tokens = [lemma.lemmatize(i) for i in tokens if not i in stopwords.words('english')]
    text = ' '.join(tokens)
    print(text)

    return text




max_features = 13463
vc = CountVectorizer(max_features=max_features, ngram_range=(1, 1))

# sample_data = ["your text data here", "more text data", ...]
# vc.fit(sample_data)


def predict(text):
    cleaned_text = clean(text)
    
    vectorized_text = vc.fit_transform([cleaned_text]).toarray()

    if vectorized_text.shape[1] < max_features:
        padding_size= max_features - vectorized_text.shape[1]
        padded_data= np.pad(vectorized_text,((0,0),(0,padding_size)),'constant')
    
    predicted_class = classifier.predict(padded_data)
    
    if predicted_class == 0:
        return "Regular"
    elif predicted_class == 1:
        return "Sarcasm"
    elif predicted_class == 2:
        return "Irony"
    elif predicted_class == 3:
        return "Figurative"

# Streamlit app
def main():
    st.title("Twitter Sentiment Analysis")

    add_selectbox = st.selectbox("How would you like to predict?", ("Sentence", "Batch File"))

    if add_selectbox == 'Sentence':
        Sentence = st.text_input("Enter Text", key="account_length")
        
        if st.button("Predict"):
            result = predict(Sentence)
            st.success(f'Sentiment: {result}')

if __name__=='__main__':
    main()