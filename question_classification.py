# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 23:45:23 2017

@author: gautam
"""
from __future__ import division
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import KFold
#from nltk.corpus import stopwords

trainfile=r'C:\Users\gautam\Downloads\kaggle\NLP\niki\trainData.txt'
testfile=r'C:\Users\gautam\Downloads\kaggle\NLP\niki\testData.txt'
resultfile=r'C:\Users\gautam\Downloads\kaggle\NLP\niki\results.txt'

## Stopwords that will occur and effect the model
stopwords=set(list(punctuation) + ['``',"''",",,,","'s","a","an","the","is","?"])
trainlabel=[]
trainData=[]

#mapping of train labels to integers
trainmap={"what" :1,"when":2,"who":3,"unknown":4,"affirmation":5}
with open(trainfile,'r') as f:
    for line in f:
        a=[]
        a=line.strip('\n').split(" ")
        trainlabel.append(a[-1])
        trainData.append(a[0:-3])

testdata=[]
testtype=[]
with open(testfile,'r') as f:
    for line in f:
        a=[]
        a=line.strip('\n').split(" ")
        testtype.append(a[0])
        testdata.append(a[1:])
 
# Processed train and test data without stopwords                
processed_data=[]
for line in trainData:
    processed_data.append([word for word in line if word not in stopwords])

processed_testdata=[]
for line in testdata:
    processed_testdata.append([word for word in line if word not in stopwords])


trainlabels=[trainmap[i] for i in trainlabel]
    
TrainingData=[' '.join(line) for line in processed_data]

# Transforming each sentence into word count vectorizer
# min_df - minimum word frequency threshold
# we can also use ngrams in this model with parameter "ngram_range=(1, 2)"
vectorizer=CountVectorizer(min_df=1)
X=vectorizer.fit_transform(TrainingData).toarray()

# Applying Multinomial (Multi-class) naive Bayes classifier for training
clf = MultinomialNB()
clf.fit(X, trainlabels)

# Cross validation scoring to check the effeciancy of model
scores = cross_validation.cross_val_score(clf,X,trainlabels, cv=3)
print "CV Mean scores %f"%np.mean(scores)

# Predicting the output of testdata
ResultLabels=[]
X_test=[]
for line in processed_testdata:
    question=' '.join(line)
    Features=vectorizer.transform([question]).toarray()
    X_test.append(Features[0])
    ResultLabels.append(clf.predict(Features)[0])
    
# mapping integers to the classes to write in the file   
labelmap={1:"what",2:"when",3:"who",4:"unknown",5:"affirmation"}
testlabels=[labelmap[i] for i in ResultLabels]

# Writing on the file
with open(resultfile,'w') as f:
    for i in range(len(testdata)):
        f.write(str(testtype[i])+" "+str(" ".join(testdata[i]))+" "+str(testlabels[i]))
        f.write('\n')
    
# Calculating Accuracy and variance on the pre-classified testdata which is same as train data    

#count=0
#for i in range(len(testdata)):
#    if ResultLabels[i]==trainlabels[i]:
#        count+=1
#Accuracy=float(count/len(testdata))
#print "Accuracy : %f"%Accuracy
#print('Variance score: %.2f' % clf.score(np.array(X_test),np.array(trainlabels[0:len(testdata)])))