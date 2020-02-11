import numpy as np
import pandas as pd
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class FakeNews:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def createModel(self):

        tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', lowercase = True, strip_accents = 'unicode')

        tfidf_train = tfidf_vectorizer.fit_transform(self.x_train) 
        tfidf_test = tfidf_vectorizer.transform(self.x_test)

        pa_classifier = PassiveAggressiveClassifier(max_iter = 200, n_iter_no_change = 10, warm_start=True)
        pa_classifier.fit(tfidf_train, self.y_train)

        y_pred = pa_classifier.predict(tfidf_test)

        correctValues, predictedValues = confusion_matrix(self.y_test, y_pred, labels=['FAKE','REAL'])
        correctFakeCount = correctValues[0]
        correctRealCount = correctValues[1]
        predictedFakeCount = predictedValues[1]
        predictedRealCount = predictedValues[0]

        print(f'Poprawne nieprawdziwe newsy: {correctFakeCount}')
        print(f'Poprawne prawdziwe newsy: {correctRealCount}')
        print(f'Przewidziane nieprawdziwe newsy: {predictedFakeCount}')
        print(f'Przewidziane prawdziwe newsy: {predictedRealCount}')

        score = accuracy_score(self.y_test, y_pred)
        print(f'Trafność: { round(score * 100, 2) }%')