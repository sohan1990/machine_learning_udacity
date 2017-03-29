#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
sort_keys = '../tools/python2_lesson14_keys.pkl'
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size= 0.3, random_state=42 )

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)

count = sum(pred)

print "Accuracy =", acc
print "Total no of POI =", count
print "total no of people in the test set =", len(features_test)
print "Accuracy for not POI prediction =", float(len(features_test)-count)/float(len(features_test))
TP = 0
for actual, prediction in zip(labels_test,pred):
    if actual == prediction and actual == 1.0:
        TP += 1
print "True Positive =",TP

print precision_score(pred, labels_test)
print recall_score(pred, labels_test)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
TN = 0
FP = 0
FN = 0
for actual, prediction in zip(true_labels,predictions):
    if actual == 1.0 and prediction == 1.0:
        TP += 1
    elif actual == 0.0 and prediction == 0.0:
        TN += 1
    elif actual == 0.0 and prediction == 1.0:
        FP +=1
    else:
        FN +=1

print "True Positive =",TP
print "True Negative =",TN
print "False Positive =",FP
print "False Negative =",FN
print precision_score(true_labels, predictions)
print recall_score(true_labels, predictions)