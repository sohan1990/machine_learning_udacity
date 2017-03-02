#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import sys
from time import time

from sklearn.naive_bayes import GaussianNB
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print 'Training time = ', round(time()-t0,3),"s"
GaussianNB(priors=None)
t1 = time()
prd = clf.predict(features_test)
print 'Testing time = ', round(time()-t1,3),"s"
count=0
for i,j in zip(prd,labels_test):
    if i==j:
        count = count + 1
accuracy = float(count)/len(labels_test)
print 'Accuracy = ',accuracy

""""
from sklearn.metrics import accuracy_score
acc = accuracy_score(prd,labels_test)
print acc
print GaussianNB
"""
#########################################################