# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 08:46:42 2021

@author: MONIORH DANAGOGO
"""

import pandas as pd
import zipfile
import time
import gzip
import dill

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from html import unescape
import spacy
from spacy.lang.en import STOP_WORDS

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
STOP_WORDS_lemma = [word.lemma_ for word in nlp(" ".join(list(STOP_WORDS)))]
STOP_WORDS_lemma = set(STOP_WORDS_lemma).union({',', '.', ';'})

class OnlinePipeline(Pipeline):
    def partial_fit(self, X, y):
        try:
            Xt = X.copy()
        except AttributeError:
            Xt = X
            
        for _, est in self.steps:
            if hasattr(est, 'partial_fit') and hasattr(est, 'predict'):
                est.partial_fit(Xt, y)
                
            if hasattr(est, 'transform'):
                Xt = est.transform(Xt)
                
        return self
    
def fit_model(func):
    def wrapper(*args, **kwargs):
        t_0 = time.time()
        model = func()
        model.fit(X_train, y_train)
        t_elapsed = time.time() - t_0
        
        print('training time: {:g}'.format(t_elapsed))
        print('training accuracy: {:g}'.format(model.score(X_train, y_train)))
        print('testing accuracy: {:g}'.format(model.score(X_test, y_test)))
        
        return model
    return wrapper

def preprocessor(doc):
        return unescape(doc).lower()
    
def tokenizer(doc):
    return [word.lemma_ for word in nlp(doc)]


@fit_model
def online_model():
    vectorizer = HashingVectorizer(preprocessor=preprocessor,
                                   tokenizer=tokenizer,
                                   ngram_range=(1, 2),
                                   stop_words=STOP_WORDS_lemma)
    
    clf = SGDClassifier(loss='log', max_iter=5)
    pipe = OnlinePipeline([
        ('vectorizer', vectorizer),
        ('classifier', clf)
        ])
    return pipe

@fit_model
def construct_model():
    vectorizer = TfidfVectorizer(preprocessor=preprocessor,
                                   tokenizer=tokenizer,
                                   ngram_range=(1, 2),
                                   stop_words=STOP_WORDS_lemma)
    
    clf = MultinomialNB()
    pipe = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', clf)
        ])
    return pipe

def serialize_model():
    model = construct_model()
    
    with gzip.open('sentiment_model.dill.gz', 'wb') as f:
        dill.dump(model, f, recurse = True)
        
if __name__ == '__main__':
    df = pd.read_csv('Sentiment_Analysis_Dataset.csv', error_bad_lines=False)
    
    X = df['SentimentText']
    y = df['Sentiment']

    splits = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_train, X_test, y_train, y_test = splits
    
    serialize_model()
    
    # model = online_model()
    # model = construct_model()
        