import pandas as pd
import numpy as np
import streamlit as st
import requests


def request_prediction(model_uri, data):
    #headers =  'Content-Type: application/json; format=pandas-records' \
    headers = {"Content-Type": "application/json", "format":"pandas-records"}

    data_json = {"Title": data}
    #print(data.values)
    #print('\n',data.index)
    #response = requests.request(
    #    method='POST', url=model_uri, json=data_json) #_json)
    
    response = requests.post(
        url=model_uri, headers=headers, json=data_json) #_json)

    print(response.text)
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

'''def suggest_tags(vectorizer, sentence_vec):
    tag_word = vectorizer.get_feature_names_out()
    tag_bool = sentence_vec.toarray()==1
    tag_sugg = tag_word[tag_bool[0]]
    return tag_sugg'''

def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    #CORTEX_URI = 'http://0.0.0.0:8890/'
    #RAY_SERVE_URI = 'http://127.0.0.1:8000/regressor'

    #api_choice = st.sidebar.selectbox(
    #    'Quelle API souhaitez vous utiliser',
    #    ['MLflow', 'Cortex', 'Ray Serve'])

    st.title('Tag suggestion to StackOverflow questions')

    title = st.text_input('Title of your StackOverflow question','')

    body = st.text_area('Ask your question','')

    predict_btn = st.button('Tag Suggestion')
    if predict_btn:
        data = title
        #data=title + ' ' + body
        pred = request_prediction(MLFLOW_URI, data).toarray()
        
        st.write(
            'Tag suggestion: {}'.format(pred))


if __name__ == '__main__':
    main()
