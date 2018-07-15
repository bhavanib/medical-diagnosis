
# coding: utf-8

# In[2]:


# importing some libraries 
import pandas as pd
import re
import nltk


# In[3]:


# importing the csv file
# make changes according to 
data=pd.read_csv('C:/Users/Vaibhav Kalakota/Downloads/mtsamples_full_text_to_csv.csv',encoding = "ISO-8859-1")


# In[4]:


# printng the 1st few rows
data.head(5)


# In[5]:


# # a.	Cardiovascular / Pulmonary
# b.	ENT - Otolaryngology
# c.	Gastroenterology
# d.	Hematology - Oncology
# e.	Nephrology
# f.	Neurology
# g.	Obstetrics / Gynecology
# h.	Urology
# i.	Ophthalmology 
# j.	Orthopedic 


# Getting the Total categories
Stype=set(data['Sample Type'])
# print(Stype)
temp=list(Stype)
# print(temp)


# deleting the types not required
temp.remove('Psychiatry / Psychology')
temp.remove('Dermatology')
# checking the list to know if data removed properly
print(len(temp))
print(temp)

SampleType=temp


# In[6]:


# Creating a dataframe with the required categories only
dataset=pd.DataFrame(columns=['Sample Type','Sample Name','Description'])
# Link of Interest=https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
count=0
r=1
for i,row in data.iterrows():
    if(row['Sample Type'] in SampleType):
#         print(list(row))
        t=row.to_frame()
#         print(type(t))
# Link of Interest-https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe/24287210
        dataset.loc[r]=list(row)
        r=r+1
print(dataset.head(5))


# In[7]:


# counting occurences of each sample type
# Link of Interet-http://cmdlinetips.com/2018/02/how-to-get-frequency-counts-of-a-column-in-pandas-dataframe/
dataset['Sample Type'].value_counts()


# In[16]:


# for preprocessing the data
import nltk
import re
from nltk.corpus import stopwords 
def cleandata(desc):
#     removing the numbers from the text
    lettersonly=re.sub("[^a-zA-Z]", " ", desc)
#     Convert to lower case, split into individual words
    words = lettersonly.lower().split()                             
    
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   


# In[17]:


# print(dataset.head())
clean=cleandata(dataset.iloc[0]['Description'])
print(clean)

# Link of Inteerest-https://stackoverflow.com/questions/33587667/extracting-all-nouns-from-a-text-file-using-nltk
# Another idea to try- trimming the strings by getting only the nouns and the verbs


# In[18]:


# Cleaning the entire dataset
# print(dataset['Description'].size)
cleandesc=[]
datasize=dataset['Description'].size
for i in range(0, datasize ):
    cleandesc.append(cleandata(dataset.iloc[i]['Description']))
# print(cleandesc)




# In[11]:


# stemming the descriptions and making a new column
# stemming the text
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

st=PorterStemmer()
s=[]
final=""
for i in cleandesc:
    words=word_tokenize(str(i))
    for j in words:
        temp=st.stem(j)
#         print(type(temp))
        final=final+" "+temp
#     print(final)
    s.append(final)
    final=" "
dataset['descwithstemming']=s
print(dataset.head(5))


# In[19]:


# making a bag of words representation for the stemmed description

from sklearn.feature_extraction.text import CountVectorizer

vector=CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None)

list1=dataset[:]['descwithstemming']

print(list1)
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.


# In[20]:


# Taking only nouns to improve the accuracy here
type(list1[0:1])
ListofStrings=list1.values.tolist()
len(ListofStrings)
tokens = nltk.word_tokenize(ListofStrings[0:1][0])
tags = nltk.pos_tag(tokens)
nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
str1 = ' '.join(nouns)
print (str1)


# In[21]:


train_f=vector.fit_transform(list1)
# Numpy arrays are easy to work with, so convert the result to an 
# array and also they are given as input to SVM
train_f=train_f.toarray()


# In[22]:


# print(vector.get_feature_names())
print(train_f.shape)


# In[135]:


# Would be useful for final prediction
# size for test, validation
from sklearn.model_selection import train_test_split
# dataset = dataset.sort_values(by=['Sample Type', 'descwithstemming'])
dataset = dataset.sort_values(by=['Sample Type', 'descwithstemming'])
y=dataset['Sample Type'] 
x=dataset['descwithstemming']
x=vector.fit_transform(x)
# Numpy arrays are easy to work with, so convert the result to an 
# array and also they are given as input to SVM
x=x.toarray()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_test[0:1])
print(y_test[0:2])
print(type(y_test))
filenameTrain='X_train.txt';
# np.save(filenameTrain, X_test)
np.savetxt(filenameTrain, X_test)
# X_test.to_csv(filenameTrain)
filenameTest='y_test.csv';
y_test.to_csv(filenameTest)


# In[24]:


# tf - idf code

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True)
tfidf_train = transformer.fit_transform(X_train).toarray();
tfidf_test = transformer.fit_transform(X_test).toarray(); 
print(tfidf_train)


# In[174]:


#trying PCA on the data
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=300)
X_train_PCA=pca.fit_transform(X_train) 
X_test_PCA=pca.fit_transform(X_test) 


# In[175]:


#PCA coverage
print(np.sum(pca.explained_variance_ratio_))


# In[58]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn import metrics
import numpy as np
X = np.random.randint(5, size=(6, 100))
ranger=(X_train.shape[0]-1)/3;
X1=X_train[0:400]
Y1=y_train[0:400]
X2=X_train[401:1201]
Y2=y_train[401:1201]

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf2 = MultinomialNB()
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)




# In[64]:


scoresNB_BOW=cross_val_score(clf, X_train, y_train, cv=5) # scores is the accuray array[5]
import pickle
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

clf1 = pickle.load(open(filename, 'rb'))

scoresTF_IDF=cross_val_score(clf,tfidf_train, y_train, cv=5) # scores is the accuray array[5]

# save the model to disk
filename = 'finalized_model2.sav'
pickle.dump(clf, open(filename, 'wb'))

clf2 = pickle.load(open(filename, 'rb'))
scoresNB_BOW_F1 = cross_val_score(clf1, X_train, y_train, cv=5, scoring='f1_macro')
scoresTF_IDF_F1=cross_val_score(clf2,tfidf_train, y_train, cv=5, scoring='f1_macro') 
scoresNB_BOW_P = cross_val_score(clf1, X_train, y_train, cv=5, scoring='precision_macro')
scoresTF_IDF_P=cross_val_score(clf2,tfidf_train, y_train, cv=5, scoring='precision_macro')
scoresNB_BOW_R = cross_val_score(clf1, X_train, y_train, cv=5, scoring='recall_macro')
scoresTF_IDF_R=cross_val_score(clf2,tfidf_train, y_train, cv=5, scoring='recall_macro')


# In[199]:


#metrics for individual classes
from sklearn.metrics import classification_report
target_names=["Cardiovascular / Pulmonary","Orthopedic","Gastroenterology","Neurology","Urology","Obstetrics / Gynecology","ENT - Otolaryngology","Hematology - Oncology","Ophthalmology","Nephrology"];
clf.fit(X2, Y2)
print(classification_report(Y1.values.tolist(), clf.predict(X1), target_names=target_names))
print(classification_report(Y1.values.tolist(), clf.predict(X1), target_names=target_names))


# In[61]:


print(scoresNB_BOW)
print(scoresTF_IDF)
print(scoresNB_BOW_F1)
print(scoresTF_IDF_F1)
print(scoresNB_BOW_P)
print(scoresTF_IDF_P)
print(scoresNB_BOW_R)
print(scoresTF_IDF_R)


# In[116]:


# Predictions 
NB_BOW_KF1=np.sum(clf.predict(X2)==Y2.values.tolist())/X2.shape[0];
NB_TFIDF_KF1=np.sum(clf2.predict(X2_tf)==Y2.values.tolist())/X2.shape[0];
print('NB for BOW KF1',NB_BOW_KF1)
print('NB for TF-IDF KF1',NB_TFIDF_KF1)

NB_BOW_KF2=np.sum(clf.predict(X4)==Y4.values.tolist())/X2.shape[0];
NB_TFIDF_KF2=np.sum(clf2.predict(X4_tf)==Y4.values.tolist())/X2.shape[0];
print('NB for BOW KF2',NB_BOW_KF2)
print('NB for TF-IDF KF2',NB_TFIDF_KF2)


# In[66]:


#SVM
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
clf = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
SVM_BOW_KF1=np.sum(clf.fit(X_train, y_train).predict(X2)==Y2.values.tolist())/X2.shape[0]
# save the model to disk
filename = 'finalized_model3.sav'
pickle.dump(clf, open(filename, 'wb'))

clf1 = pickle.load(open(filename, 'rb'))

SVM_TFIDF_KF1=np.sum(clf.fit(tfidf_train, y_train).predict(X2_tf)==Y2.values.tolist())/X2.shape[0]

# save the model to disk
filename = 'finalized_model4.sav'
pickle.dump(clf, open(filename, 'wb'))

clf2 = pickle.load(open(filename, 'rb'))

SVM_BOW_KF2=np.sum(clf1.fit(X_train, y_train).predict(X4)==Y4.values.tolist())/X2.shape[0]
SVM_TFIDF_KF1=np.sum(clf2.fit(tfidf_train, y_train).predict(X4_tf)==Y4.values.tolist())/X2.shape[0]
scoresSVM_BOW=cross_val_score(clf1, X_train, y_train, cv=5) # for SVM - BOW
scoresSVMTF_IDF=cross_val_score(clf2,tfidf_train, y_train, cv=5) # for SVM - TF-IDF
scoresSVM_BOW_F1 = cross_val_score(clf1, X_train, y_train, cv=5, scoring='f1_macro')
scoresSVMTF_IDF_F1=cross_val_score(clf2,tfidf_train, y_train, cv=5, scoring='f1_macro') 
scoresSVM_BOW_P = cross_val_score(clf1, X_train, y_train, cv=5, scoring='precision_macro')
scoresSVMTF_IDF_P=cross_val_score(clf2,tfidf_train, y_train, cv=5, scoring='precision_macro')
scoresSVM_BOW_R = cross_val_score(clf1, X_train, y_train, cv=5, scoring='recall_macro')
scoresSVMTF_IDF_R=cross_val_score(clf2,tfidf_train, y_train, cv=5, scoring='recall_macro')

print('SVM for BOW KF1',SVM_BOW_KF1)
print('SVM for TF-IDF KF1',SVM_TFIDF_KF1)
print('SVM for BOW KF2',SVM_BOW_KF2)
print('SVM for TF-IDF KF2',SVM_TFIDF_KF1)


# In[182]:


# USing Normalizer to processes the inputs i.e Negative values-Postive values -----> (0,1)
from sklearn import preprocessing
binarizer = preprocessing.Binarizer().fit(X_train_PCA)
X_train_PCA_NB=binarizer.transform(X_train_PCA)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train_PCA_NB_MMS=scaler.fit(X_train_PCA).transform(X_train_PCA)

X_train_PCA_NB_normalized = preprocessing.normalize(X_train_PCA, norm='l2')


# In[195]:


clf_NB_PCA = MultinomialNB()
scoresNB_BOW_PCA=cross_val_score(clf_NB_PCA, X_train_PCA_NB, y_train, cv=10) # for SVM - BOW
scoresSVMTF_IDF_PCA=cross_val_score(clf2,tfidf_train, y_train, cv=10) # for SVM - TF-IDF
scoresNB_BOW_F1_PCA = cross_val_score(clf1, X_train_PCA_NB, y_train, cv=10, scoring='f1_macro')
scoresTF_IDF_F1_PCA=cross_val_score(clf2,tfidf_train, y_train, cv=10, scoring='f1_macro') 
scoresNB_BOW_P_PCA = cross_val_score(clf1, X_train_PCA_NB, y_train, cv=10, scoring='precision_macro')
scoresTF_IDF_P_PCA=cross_val_score(clf2,tfidf_train, y_train, cv=10, scoring='precision_macro')
scoresNB_BOW_R_PCA = cross_val_score(clf1, X_train_PCA_NB, y_train, cv=10, scoring='recall_macro')
scoresTF_IDF_R_PCA=cross_val_score(clf2,tfidf_train, y_train, cv=10, scoring='recall_macro')


# In[192]:


print(scoresNB_BOW_PCA)
print(scoresSVMTF_IDF_PCA)
print(scoresNB_BOW_F1_PCA)
print(scoresTF_IDF_F1_PCA)
print(scoresNB_BOW_P_PCA)
print(scoresTF_IDF_P_PCA)
print(scoresNB_BOW_R_PCA)
print(scoresTF_IDF_R_PCA)


# In[194]:


#SVM after PCA
# print(X_test_PCA.shape)
# print(X_train_PCA.shape)
scoresSVM_BOW_PCA=cross_val_score(clf1, X_train_PCA, y_train, cv=10) # for SVM - BOW
scoresSVMTF_IDF=cross_val_score(clf2,tfidf_train, y_train, cv=10) # for SVM - TF-IDF
scoresSVM_BOW_F1 = cross_val_score(clf1, X_train_PCA, y_train, cv=10, scoring='f1_macro')
scoresSVMTF_IDF_F1=cross_val_score(clf2,tfidf_train, y_train, cv=10, scoring='f1_macro') 
scoresSVM_BOW_P = cross_val_score(clf1, X_train_PCA, y_train, cv=10, scoring='precision_macro')
scoresSVMTF_IDF_P=cross_val_score(clf2,tfidf_train, y_train, cv=10, scoring='precision_macro')
scoresSVM_BOW_R = cross_val_score(clf1, X_train_PCA, y_train, cv=10, scoring='recall_macro')
scoresSVMTF_IDF_R=cross_val_score(clf2,tfidf_train, y_train, cv=10, scoring='recall_macro')


# In[190]:


print(scoresSVM_BOW_PCA)


# In[186]:


print(scoresSVM_BOW)
print(scoresSVMTF_IDF)
print(scoresSVM_BOW_F1)
print(scoresSVMTF_IDF_F1)
print(scoresSVM_BOW_P)
print(scoresSVMTF_IDF_P)
print(scoresSVM_BOW_R)
print(scoresSVMTF_IDF_R)


# In[35]:


# Neural Networks
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(20, 20), random_state=1)
clf.fit(X1, Y1)
print('NN for BOW KF1',np.sum(clf.fit(X1, Y1).predict(X2)==Y2.values.tolist())/X2.shape[0])


# In[193]:


#testing
# x_test=pd.read_csv('C:/Users/Vaibhav Kalakota/Downloads/y_train.csv',encoding = "ISO-8859-1")
# outfile='y_train.csv';
# x_test=np.load(outfile)
x_testOutput=np.loadtxt("C:/Users/Vaibhav Kalakota/Downloads/X_train.txt")
y_testOuput=pd.read_csv('C:/Users/Vaibhav Kalakota/Downloads/y_test.csv',encoding = "ISO-8859-1")
# clf2.fit(X1_tf, Y1)
# # clf3.fit(X1_pca,Y1)
# MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
# print(clf.predict(x_testOutput[0:10]))
# print(y_testOuput[0:10])
# print(np.sum(clf.predict(x_test)==y_test))
outPut=clf.predict(x_testOutput);
print(outPut[0:10])
print(y_testOuput[0:10])
print(np.sum(clf.predict(x_testOutput)==y_testOuput.values.tolist()))
print('NB for PCA BOW KF1',clf.score(x_testOutput,y_testOuput))

