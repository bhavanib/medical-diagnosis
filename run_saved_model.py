import pickle
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

dataset = pd.read_csv('test_file_y.csv', delimiter=',')

y = dataset['SampleType']
X = np.loadtxt('test_file_x.txt', dtype=int)

#tf - idf code
transformer = TfidfTransformer(smooth_idf=False)
tfidf_test = transformer.fit_transform(X).toarray()

clf = pickle.load(open('savemodel-NB-TF.sav', 'rb'))
result = clf.score(X, y)
print ('NB TF: ', result)

clf = pickle.load(open('savemodel-NB-TFIDF.sav', 'rb'))
result = clf.score(tfidf_test, y)
print ('NB TFIDF: ', result)

clf = pickle.load(open('savemodel-SVM-TF.sav', 'rb'))
result = clf.score(X, y)
print ('SVM TF: ', result)

clf = pickle.load(open('savemodel-SVM-TFIDF.sav', 'rb'))
result = clf.score(tfidf_test, y)
print ('SVM TFIDF: ', result)
