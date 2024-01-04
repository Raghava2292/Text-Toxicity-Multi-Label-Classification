# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:01:20 2023

@author: harivars
"""

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import codecs
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
import pickle
from keras.models import load_model
from bs4 import BeautifulSoup
warnings.filterwarnings("ignore")
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import streamlit as st
st.title('Toxicity analysis of a sentence')
predict_option = st.radio('Select One Option:', ('Single Prediction', 'Dataset Prediction'))


def remove_contractions(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def remove_punctuations(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def clean_sentences(sentence):
    sentence = str(sentence)
    sentence= re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = remove_contractions(sentence)
    sentence = remove_punctuations(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    stop_words = set(stopwords.words('english'))
    stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in  stopwords.words('english'))
    return sentence.strip()

def tokenize(sentence):
    MAX_SEQUENCE_LENGTH = 400
    #MAX_NB_WORDS = 50000
    with open('tokenizer_lstm.pickle', 'rb') as handle:
                    tokenizer = pickle.load(handle)
    test_sequences = tokenizer.texts_to_sequences([sentence])
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return test_data

def model_predict(model, test_data):
    
    if model == 'LSTM':
        model=load_model('LSTM_toxic_prediction_model3.h5')
    elif model == 'Bidirectional LSTM':
        model = load_model('BiLSTM_toxic_prediction_model3.h5')
    prediction=model.predict(test_data)    
    return prediction

def get_prediction(model, sentence):
    test_data=tokenize(sentence)
    if model == 'LSTM':
        model=load_model('LSTM_toxic_prediction_model3.h5')
    elif model == 'Bidirectional LSTM':
        model = load_model('BiLSTM_toxic_prediction_model3.h5')
    
    predicted_array=model.predict(test_data)
    predicted_values={'Toxic':round(predicted_array[0][0]),'Severe Toxic':round(predicted_array[0][1]), 'Obscene':round(predicted_array[0][2]), 'Threat':round(predicted_array[0][3]), 'Insult':round(predicted_array[0][4]), 'Hatred':round(predicted_array[0][5])}

    result_list=[]
    for key in predicted_values:
        if(predicted_values[key]==1.0):
            result_list.append(key)
    result = ','.join(result_list)
    if result == 'Toxic,Severe Toxic,Obscene,Insult':
        result_cat = 'Intense Malicious Disparagement'
        toxic_per = 90
    elif result == 'Toxic,Obscene,Insult,Hatred':
        result_cat = 'Venomous Reprehension'
        toxic_per = 80
    elif result == 'Toxic,Obscene,Insult':
        result_cat = 'Malicious Indecency'
        toxic_per = 70
    elif result == 'Toxic,Obscene':
        result_cat = 'Noxious'
        toxic_per = 60
    elif result == 'Toxic,Threat':
        result_cat = 'Menacing'
        toxic_per = 50
    elif result == 'Toxic,Insult':
        result_cat = 'Offensive'
        toxic_per = 50
    elif result == 'Toxic,Obscene,Threat,Insult,Hatred':
        result_cat = 'Malevolent Vulgarity'
        toxic_per = 95
    elif result == 'Toxic,Severe Toxic,Obscene':
        result_cat = 'Intense Contamination'
        toxic_per = 80
    elif result == 'Toxic,Obscene,Threat,Insult':
        result_cat = 'Dangerous Provocation'
        toxic_per = 85
    elif result == 'Toxic,Severe Toxic,Obscene,Insult,Hatred':
        result_cat = 'Excessive Malevolence'
        toxic_per = 90
    elif result == 'Toxic,Severe Toxic,Obscene,Threat,Insult,Hatred':
        result_cat = 'Overwhelming Hostility'
        toxic_per = 95
    elif result == 'Toxic,Insult,Hatred':
        result_cat = 'Hostile Disdain'
        toxic_per = 70
    elif result == 'Toxic,Hatred':
        result_cat = 'Virulent Hostility'
        toxic_per = 60
    elif result == 'Obscene,Insult':
        result_cat = 'Vulgar Reproach'
        toxic_per = 70
    elif result == 'Toxic,Severe Toxic,Obscene,Hatred':
        result_cat = 'Excessive Hostility'
        toxic_per = 85
    elif result == 'Toxic,Severe Toxic,Obscene,Threat,Insult':
        result_cat = 'Overwhelming Menace'
        toxic_per = 90
    elif result == 'Toxic,Obscene,Hatred':
        result_cat = 'Harmful Repugnance'
        toxic_per = 75
    elif result == 'Obscene,Insult,Hatred':
        result_cat = 'Indecent Contempt'
        toxic_per = 80
    elif result == 'Toxic,Severe Toxic':
        result_cat = 'Intense Toxicity'
        toxic_per = 60
    elif result == 'Toxic,Severe Toxic,Insult':
        result_cat = 'Severe Disdain'
        toxic_per = 70
    elif result == 'Toxic,Severe Toxic,Hatred':
        result_cat = 'Intense Malevolence'
        toxic_per = 80
    elif result == 'Obscene,Threat':
        result_cat = 'Offensive Threat'
        toxic_per = 50
    elif result == 'Toxic,Threat,Insult':
        result_cat = 'Hazardous Assault'
        toxic_per = 60
    elif result == 'Insult,Hatred':
        result_cat = 'Offensive Animosity'
        toxic_per = 60
    elif result == 'Toxic,Severe Toxic,Obscene,Threat':
        result_cat = 'Overwhelming Peril'
        toxic_per = 90
    elif result == 'Toxic,Severe Toxic,Insult,Hatred':
        result_cat = 'Intense Hostility'
        toxic_per = 90
    elif result == 'Toxic,Obscene,Threat':
        result_cat = 'Hazardous Provocation'
        toxic_per = 80
    elif result == 'Toxic,Severe Toxic,Threat':
        result_cat = 'Severe Menace'
        toxic_per = 85
    elif result == 'Threat,Insult':
        result_cat = 'Threatening Disdain'
        toxic_per = 50
    elif result == 'Toxic,Threat,Insult,Hatred':
        result_cat = 'Perilous Animosity'
        toxic_per = 75
    elif result == 'Toxic,Threat,Hatred':
        result_cat = 'Dangerous Hostility'
        toxic_per = 80
    elif result == 'Obscene,Threat,Insult':
        result_cat = 'Indecent Threat'
        toxic_per = 70
    elif result == 'Toxic,Severe Toxic,Threat,Insult':
        result_cat = 'Intense Disparagement'
        toxic_per = 85
    elif result == 'Toxic,Severe Toxic,Threat,Hatred':
        result_cat = 'Overwhelming Malevolence'
        toxic_per = 90
    elif result == 'Obscene,Hatred':
        result_cat = 'Indecent Animosity'
        toxic_per = 60

    elif result == 'Toxic':
        result_cat = 'Toxic'
        toxic_per = 40
    elif result == 'Severe Toxic':
        result_cat = 'Severe Toxic'
        toxic_per = 50
    elif result == 'Obscene':
        result_cat = 'Obscene'
        toxic_per = 50
    elif result == 'Threat':
        result_cat = 'Threat'
        toxic_per = 50
    elif result == 'Insult':
        result_cat = 'Insult'
        toxic_per = 50
    elif result == 'Hatred':
        result_cat = 'Hatred'
        toxic_per = 60
    else:
        result = 'General'
        result_cat = 'Harmless or Positive Commentary'
        toxic_per = 0   
        
    output_list = result.split(',')
    output = ', '.join(output_list)
    
    return output, result_cat, toxic_per, predicted_array[0]

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()

def show_barplot(data, title=None):
    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(plt.bar(['Toxic', 'Severe_Toxic', 'Obscene', 'Threat', 'Insult', 'Hatred'], list(data), width=0.4 ))
    plt.show()
    st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

## Single prediction

if predict_option == 'Single Prediction':
    sentence = st.text_input('Enter the statement here: ')
    model = st.radio('Select a model:', ('LSTM', 'Bidirectional LSTM'))
    if st.button('Predict'):
        clear_text=clean_sentences(sentence)
        
        
        prediction, prediction_cat, prediction_per, prediction_arr = get_prediction(model, clear_text)
        
    
        st.markdown(f'The categories to which the given statement belongs to are - **:orange[{prediction}]**')
        st.markdown(f'Combining these individual categories will make the sentence as - **:orange[{prediction_cat}]**')
        st.markdown(f'The toxicity percentage of this sentence can be interpreted as - **:orange[{prediction_per}%]**')

## Dataset prediction

elif predict_option == 'Dataset Prediction':
    st.subheader('Upload the dataset')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None: 
        df = pd.read_csv(uploaded_file)
        df['clean_text'] = df.comment_text.apply(lambda x: clean_sentences(x))
        
        model = st.radio('Select a model:', ('LSTM', 'Bidirectional LSTM'))
        
        st.subheader('Get the names of people present in the file')
        if st.button('Get names'):
            nlp = spacy.load('en_core_web_lg')
            nlp.max_length = 37000000
            block = df['clean_text'].values
            block = ' '.join(block)
            doc_block = nlp(block)
            ner_categories = ['PERSON']
            entities = []
            for ent in doc_block.ents:
                if ent.label_ in ner_categories:
                    entities.append((ent.text).capitalize())
            entities = list(set(entities))
            if len(entities) != 0:
                st.write(', '.join(entities))
            else:
                st.write('No named entities in the text data')
                    
        st.subheader('Get toxicity predictions of the text in the file:')
        if st.button('Predict'):
            df['predictions'] = df['clean_text'].apply(lambda x: get_prediction(model, x))
            df['result'] = df['predictions'].apply(lambda x: x[0])
            df['result_category'] = df['predictions'].apply(lambda x: x[1])
            df['toxicity_percentage'] = df['predictions'].apply(lambda x: x[2])
            df.drop('predictions', axis=1, inplace=True)
            st.dataframe(df)
            
            st.download_button('Download Predictions', data = df.to_csv().encode('utf-8'), file_name=f'Precited Toxicities.csv', mime='text/csv')
        
        
        st.subheader('Get a display of the different toxic words present in the file')
        if st.button('Get toxic words'):
            df['predictions'] = df['clean_text'].apply(lambda x: get_prediction(model, x))
            df['result'] = df['predictions'].apply(lambda x: x[0])
            toxic_data = df[df['result'] != 'General']['clean_text']
            show_wordcloud(toxic_data)
        
    