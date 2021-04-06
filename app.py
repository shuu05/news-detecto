import json
from feature import *
import joblib
from flask import Flask, render_template, jsonify, abort, request
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def get_all_query(title, author, text):
    total = title + author + text
    total = [total]
    return total


def cleaning_and_processing(sentence):
    filter_sentence = ''
    lemmatizer = WordNetLemmatizer()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    words = nltk.word_tokenize(sentence)
    words = [w for w in words if w in stopwords]
    for word in words:
        filter_sentence = filter_sentence + ' ' + \
            str(lemmatizer.lemmatize(word)).lower()
    return filter_sentence


pipeline = joblib.load('pipeline.sav')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api', methods=['POST'])
def get_delay():
    result = request.form
    query_title = result['title']
    query_author = result['author']
    query_text = result['maintext']
    print(query_text)
    query = get_all_query(query_title, query_author, query_text)
    user_input = {'query': query}
    pred = pipeline.predict(query)
    print(pred)
    dic = {'1': 'Real', '0': 'Fake'}
    return f'<html><body><h1>{dic[pred[0]]}</h1> <form action="/"> <button type="submit">back </button> </form></body></html>'


if __name__ == '__main__':
    app.run(debug=True)
