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
    sentence = re.sub(r'[^\w\s]', '', s)
    words = nltk.word_tokenize(sentence)
    words = [w for w in words if w in stopwords]
    for word in words:
        filter_sentence = filter_sentence + ' ' + \
            str(lemmatizer.lemmatize(word)).lower()
    return filter_sentence
