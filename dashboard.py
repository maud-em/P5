import pandas as pd
import numpy as np
import streamlit as st
import joblib
from custom_transformer import textNormalizer
   
my_tags = pd.read_csv('tags.csv')
tags = my_tags['0'].values

def suggest_tags(tags, sentence_vec):
    tag_bool = sentence_vec==1
    tag_sugg = np.array(tags)[tag_bool[0]]
    return tag_sugg

def supervised_prediction(data, pkl_model = 'tfidf_model.pkl'):
    multitag_model = open(pkl_model,'rb')
    clf = joblib.load(multitag_model)
    my_prediction = clf.predict(data)
    #print(my_prediction)
    #tag_bool = my_prediction.toarray()==1
    tag_sugg = suggest_tags(tags,my_prediction)
    return tag_sugg


def main():
    st.title('Tag suggestion to StackOverflow questions')

    title = st.text_input('Title of your StackOverflow question','')

    body = st.text_area('Ask your question','')

    predict_btn = st.button('Tag Suggestion')
    if predict_btn:
        data = [title]
        #data=title + ' ' + body
        pred = supervised_prediction(data)
        st.write('Supervised model: LinearSVC classifier in OnevsRest Wrapper')
        st.write(pred)
        data = [title + ' ' + body]
        pred = supervised_prediction(data,'tfidf_nltk_model.pkl')

if __name__ == '__main__':
    main()

