#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


### data load and pre-process
def data_load_process():
    # Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # Remove outliers
    outliers = ['TOTAL']
    for outlier in outliers:
        data_dict.pop(outlier, 0)

    #print data_dict

    return data_dict


### classifier function
def classifier_list():

    clf_list = []
    # NAIVE BAYES
    from sklearn.naive_bayes import GaussianNB
    clf_NB = GaussianNB()
    clf_NB_params = {}
    clf_list.append((clf_NB, clf_NB_params))

    # SVM
    from sklearn.svm import LinearSVC
    clf_svm = LinearSVC()
    clf_svm_params = {"C": [1, 5, 10, 15, 20],
                     "tol": [0.1, 0.01, 0.001, 0.0001],
                     "dual": [False]}
    clf_list.append((clf_svm, clf_svm_params))

    # DECISION TREE
    from sklearn.tree import DecisionTreeClassifier
    clf_tree = DecisionTreeClassifier()
    clf_tree_params = {"min_samples_split": [2,5,10,15,20],
                      "criterion": ["gini", "entropy"]}
    clf_list.append((clf_tree, clf_tree_params))

    # ADABOOST
    from sklearn.ensemble import AdaBoostClassifier
    clf_ada = AdaBoostClassifier()
    clf_ada_params = {"n_estimators": [20, 30, 40, 50],
                      #"learning_rate": [0.1, 0.5, 1, 2, 5],
                      "algorithm": ["SAMME", "SAMME.R"]}
    clf_list.append((clf_ada, clf_ada_params))

    # RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    clf_rforest = RandomForestClassifier()
    clf_rforest_params = {"n_estimators": [20, 30, 40, 50],
                          "criterion": ["gini", "entropy"],
                          "min_samples_split": [2, 5, 10, 15, 20]}
    clf_list.append((clf_rforest, clf_rforest_params))

    # LOGISTIC REGRESSION
    from sklearn.linear_model import LogisticRegression
    clf_logreg = LogisticRegression()
    clf_logreg_params = {"C": [0.1, 1, 10, 50, 100],
                         "tol": [0.1, 0.01, 0.001, 0.0001],}
    clf_list.append((clf_logreg, clf_logreg_params))

    return clf_list


### Applying gridsearch to the classifier list
def classify(clf_list, features_train, labels_train):

    # testing gridsearchcv
    from sklearn.model_selection import GridSearchCV
    best_clf_list = []

    for classifier, params in clf_list:

        clf_rev = GridSearchCV(classifier, params)
        clf_rev.fit(features_train, labels_train)
        try:
            best_clf_list.append(clf_rev.best_estimator_)
        except:
            print "no best estimator available for this classifier"

    return best_clf_list

### Evaluate the accuracy, recall and precision of the classifiers
def evaluate(clf_list, features_test, labels_test):

    from sklearn.metrics import recall_score, precision_score, f1_score

    new_list = []
    for clf in clf_list:
        pred = clf.predict(features_test)
        recall = recall_score(labels_test, pred)
        precision = precision_score(labels_test, pred)
        f1 = f1_score(labels_test, pred)
        new_list.append((clf, recall, precision, f1))

    return new_list

### sort the classifier list according to their f1-score
def sort_clf(clf_list):

    clf_list_sorted = sorted(clf_list, key = lambda x:x[3], reverse=True)

    return clf_list_sorted


### Task 1: Select what features you'll use.

poi = ['poi']

financial_features = ['salary',
                      'deferral_payments',
                      'total_payments',
                      'loan_advances',
                      'bonus',
                      'restricted_stock_deferred',
                      'deferred_income',
                      'total_stock_value',
                      'expenses',
                      'exercised_stock_options',
                      'other',
                      'long_term_incentive',
                      'restricted_stock',
                      'director_fees']

email_features = ['to_messages',
                  'from_poi_to_this_person',
                  'from_messages',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi']


features_list = poi + financial_features + email_features
#print features_list


### Task 3: Create new feature(s)

new_features = []

### Store to my_data set for easy export below.
data_dict = data_load_process()
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# create the classifier list
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()


### Task 5: Data processing

from sklearn.decomposition import PCA

def preprocessing_pca(features_train, features_test):
    pca = PCA(n_components=5, svd_solver='randomized').fit(features_train)
    features_train_pca = pca.transform(features_train)
    features_test_pca = pca.transform(features_test)

    return features_train_pca, features_test_pca


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
features_train, features_test = preprocessing_pca(features_train, features_test)




### Task 6: Applying classifier

# creatiing the classifier list
clf_list = classifier_list()
# getting the classifiers with their best parameters
clf_list = classify(clf_list, features_train, labels_train)

# Evaluate the precision, recall and f1 scores  of the classifiers
clf_list = evaluate(clf_list, features_test, labels_test)
clf_list_sorted = sort_clf(clf_list)
print clf_list_sorted
clf =  clf_list_sorted[1][0]
"""


###testing pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
#classifier
clf = LogisticRegression()
clf_params = {"clf__C": [0.1, 1, 10, 50, 100], "clf__tol": [0.1, 0.01, 0.001, 0.0001]}

# pre-processing
from sklearn.decomposition import PCA
pca = PCA()
pca_params = {'pca__n_components': [5, 10, 20]}
# pipeline
clf_pipeline = Pipeline([('pca', pca),('logistic_regression',clf)])
clf_params.update(pca_params)
print clf_params
#clf_pipeline.set_params(clf_params)

#gridsearch
from sklearn.model_selection import GridSearchCV
clf_pipeline = GridSearchCV(clf_pipeline, clf_params)
#clf_pipeline = GridSearchCV(clf_pipeline)
print clf_pipeline.get_params().keys()
clf_pipeline.fit(features_train, labels_train)
clf = clf_pipeline.best_estimator_
clf = evaluate(clf, features_test, labels_test)
print clf
"""
### Task 7: Dump your classifier, dataset, and features_list

dump_classifier_and_data(clf, my_dataset, features_list)