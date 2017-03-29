#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




### Task 1: Select what features you'll use.

poi = ['poi']

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

#email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
#                 'shared_receipt_with_poi']


features_list = poi + financial_features
#features_list = ['poi', 'salary', 'bonus']
print features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#print data_dict





### Task 2: Remove outliers

outliers = ['TOTAL']
for outlier in outliers:
    data_dict.pop(outlier,0)




### Task 3: Create new feature(s)

new_features = []

### Store to my_data set for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)






### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()






### Task 5: Data processing

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.decomposition import PCA

def preprocessing_pca(features_train, features_test):
    pca = PCA(n_components=5, svd_solver='randomized').fit(features_train)
    features_train_pca = pca.transform(features_train)
    features_test_pca = pca.transform(features_test)

    return features_train_pca, features_test_pca



### Task 6: Apply classifier

from sklearn.model_selection import KFold
kf = KFold(n_splits=100)
for train_indices, test_indices in kf.split(features):
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]

    features_train, features_test = preprocessing_pca(features_train, features_test)

    clf.fit(features_train, labels_train)


pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print acc







### Task 7: Dump your classifier, dataset, and features_list

dump_classifier_and_data(clf, my_dataset, features_list)