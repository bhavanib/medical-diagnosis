import re
import warnings

import pickle
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')

data = pd.read_csv('../clampdata.csv', encoding="ISO-8859-1")

# Getting the Total categories
Stype = set(data['SampleType'])
temp = list(Stype)


# deleting the types not required
temp.remove('Psychiatry / Psychology')
temp.remove('Dermatology')

SampleType = temp

# Creating a dataframe with the required categories only
dataset = pd.DataFrame(columns=['SampleType', 'SampleName', 'Problem', 'Drug', 'Treatment', 'Test'])
count = 0
r = 1
for i, row in data.iterrows():
    if row['SampleType'] in SampleType:
        t = row.to_frame()
        dataset.loc[r] = list(row)
        r = r + 1

# counting occurences of each sample type
dataset['SampleType'].value_counts()


# for preprocessing the data
# removing the numbers from the text
# Convert to lower case, split into individual words
# a list, so convert the stop words to a set
# 5. Remove stop words
# 6. Join the words back into one string separated by space,
# and return the result.
def cleandata(desc):
    lettersonly = re.sub("[^a-zA-Z]", " ", desc)
    words = lettersonly.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]
    return " ".join(meaningful_words)


# Cleaning the entire dataset
cleandesc = []
dropindices = []
datasize = dataset['Test'].size
for i in range(0, datasize):
    prob = dataset.iloc[i]['Problem'] if not isinstance(dataset.iloc[i]['Problem'], float) else ""
    drug = dataset.iloc[i]['Drug'] if not isinstance(dataset.iloc[i]['Drug'], float) else ""
    trea = dataset.iloc[i]['Treatment'] if not isinstance(dataset.iloc[i]['Treatment'], float) else ""
    test = dataset.iloc[i]['Test'] if not isinstance(dataset.iloc[i]['Test'], float) else ""
    cleandesc.append(cleandata(prob))


# TODO add indices to the list when running for single columns
# for i in dropindices:
#    dataset.drop(dataset.index[i])


# stemming the descriptions and making a new column
# stemming the text
lemma = WordNetLemmatizer()
s = []
final = ''
for i in cleandesc:
    words = word_tokenize(str(i))
    for j in words:
        temp = lemma.lemmatize(j)
        final = final + " " + temp
    s.append(final)
    final = ''
dataset = dataset.drop(dataset.index[dropindices])
dataset['descwithstemming'] = s

# making a bag of words representation for the stemmed description
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.

# Would be useful for final prediction
# size for test, validation
vector = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=None)
# dataset = dataset.sort_values(by=['SampleType', 'descwithstemming'])
y = dataset['SampleType']
x = dataset['descwithstemming']

x = vector.fit_transform(x)
print (x.shape)

# remove features with zero variance
selector = VarianceThreshold(threshold=0)
x = selector.fit_transform(x)
print (x.shape)

# select best 300 features using chi2 stats
x = SelectKBest(chi2, k=300).fit_transform(x, y)
print (x.shape)

# Numpy arrays are easy to work with, so convert the result to an
# array and also they are given as input to SVM
x = x.toarray()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

test_dataset = pd.DataFrame(columns=['SampleType'])
test_dataset['SampleType'] = y_test
test_dataset.to_csv('test_file_y.csv', sep=',', encoding='utf-8')
np.savetxt('test_file_x.txt', X_test, fmt='%d')

# tf - idf code
transformer = TfidfTransformer(smooth_idf=False)
tfidf_train = transformer.fit_transform(X_train).toarray()
tfidf_test = transformer.fit_transform(X_test).toarray()

# target names required for per class data
target_names = ["Cardiovascular / Pulmonary", "Orthopedic", "Gastroenterology", "Neurology", "Urology",
                "Obstetrics / Gynecology", "ENT - Otolaryngology", "Hematology - Oncology", "Ophthalmology", "Nephrology"]

# NB
clf = MultinomialNB()
scoresNB_BOW = cross_val_score(clf, X_train, y_train, cv=10)
scoresNB_BOW_F1 = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1_macro')
scoresNB_BOW_P = cross_val_score(clf, X_train, y_train, cv=10, scoring='precision_macro')
scoresNB_BOW_R = cross_val_score(clf, X_train, y_train, cv=10, scoring='recall_macro')
clf.fit(X_train, y_train)
per_class_NB_BOW = classification_report(y_train.values.tolist(), clf.predict(X_train), target_names=target_names)
print ('\nscores NB TF -> \nAccuracy: ', scoresNB_BOW, '\nF1: ', scoresNB_BOW_F1, '\nPrecision: ', scoresNB_BOW_P,
       '\nRecall: ', scoresNB_BOW_R, '\nPer Class Metrics:\n', per_class_NB_BOW)
pickle.dump(clf, open('savemodel-NB-TF.sav', 'wb'))

clf = MultinomialNB()
scoresTF_IDF = cross_val_score(clf, tfidf_train, y_train, cv=10)
scoresTF_IDF_F1 = cross_val_score(clf, tfidf_train, y_train, cv=10, scoring='f1_macro')
scoresTF_IDF_P = cross_val_score(clf, tfidf_train, y_train, cv=10, scoring='precision_macro')
scoresTF_IDF_R = cross_val_score(clf, tfidf_train, y_train, cv=10, scoring='recall_macro')
clf.fit(tfidf_train, y_train)
per_class_TF_IDF = classification_report(y_train.values.tolist(), clf.predict(tfidf_train), target_names=target_names)
print ('\nscores NB TFIDF -> \nAccuracy: ', scoresTF_IDF_F1, '\nF1: ', scoresTF_IDF_F1, '\nPrecision: ',
       scoresTF_IDF_P, '\nRecall: ', scoresTF_IDF_R, '\nPer Class Metrics:\n', per_class_TF_IDF)
pickle.dump(clf, open('savemodel-NB-TFIDF.sav', 'wb'))

# SVM
clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
scoresSVM_BOW = cross_val_score(clf, X_train, y_train, cv=10)
scoresSVM_BOW_F1 = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1_macro')
scoresSVM_BOW_P = cross_val_score(clf, X_train, y_train, cv=10, scoring='precision_macro')
scoresSVM_BOW_R = cross_val_score(clf, X_train, y_train, cv=10, scoring='recall_macro')
clf.fit(X_train, y_train)
per_class_SVM_BOW = classification_report(y_train.values.tolist(), clf.predict(X_train), target_names=target_names)
print ('\nscores SVM TF -> \nAccuracy: ', scoresSVM_BOW, '\nF1: ', scoresSVM_BOW_F1, '\nPrecision: ',
       scoresSVM_BOW_P, '\nRecall: ', scoresSVM_BOW_R, '\nPer Class Metrics:\n', per_class_SVM_BOW)
pickle.dump(clf, open('savemodel-SVM-TF.sav', 'wb'))

clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
scoresSVMTF_IDF = cross_val_score(clf, tfidf_train, y_train, cv=10)
scoresSVMTF_IDF_F1 = cross_val_score(clf, tfidf_train, y_train, cv=10, scoring='f1_macro')
scoresSVMTF_IDF_P = cross_val_score(clf, tfidf_train, y_train, cv=10, scoring='precision_macro')
scoresSVMTF_IDF_R = cross_val_score(clf, tfidf_train, y_train, cv=10, scoring='recall_macro')
clf.fit(X_train, y_train)
per_class_SVMTF_IDF = classification_report(y_train.values.tolist(), clf.predict(tfidf_train), target_names=target_names)
print ('\nscores SVM TFIDF -> \nAccuracy: ', scoresSVMTF_IDF, '\nF1: ', scoresSVMTF_IDF_F1, '\nPrecision: ',
       scoresSVMTF_IDF_P, '\nRecall: ', scoresSVMTF_IDF_R, '\nPer Class Metrics:\n', per_class_SVMTF_IDF)
pickle.dump(clf, open('savemodel-SVM-TFIDF.sav', 'wb'))
