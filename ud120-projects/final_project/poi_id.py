#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

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
    #from sklearn.naive_bayes import GaussianNB
    #clf_NB = GaussianNB()
    #clf_NB_params = {}
    #clf_list.append((clf_NB, clf_NB_params))

    # SVM
    clf_svm = LinearSVC()
    clf_svm_params = {"C": [1, 5, 10, 15, 20],
                     "tol": [0.1, 0.01, 0.001, 0.0001],
                     "dual": [False]}
    clf_list.append((clf_svm, clf_svm_params))

    # DECISION TREE
    clf_tree = DecisionTreeClassifier()
    clf_tree_params = {"min_samples_split": [2,5,10,15,20],
                      "criterion": ["gini", "entropy"]}
    clf_list.append((clf_tree, clf_tree_params))

    # ADABOOST
    clf_ada = AdaBoostClassifier()
    clf_ada_params = {"n_estimators": [20, 30, 40, 50],
                      #"learning_rate": [0.1, 0.5, 1, 2, 5],
                      "algorithm": ["SAMME", "SAMME.R"]}
    clf_list.append((clf_ada, clf_ada_params))

    # RANDOM FOREST
    clf_rforest = RandomForestClassifier()
    clf_rforest_params = {"n_estimators": [20, 30, 40, 50],
                          "criterion": ["gini", "entropy"],
                          "min_samples_split": [2, 5, 10, 15, 20]}
    clf_list.append((clf_rforest, clf_rforest_params))

    # LOGISTIC REGRESSION
    clf_logreg = LogisticRegression()
    clf_logreg_params = {"C": [0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20],
                         "tol": [10**-1, 10**-5, 10**-10]}
    clf_list.append((clf_logreg, clf_logreg_params))

    return clf_list



### Applying gridsearch to the classifier list
def classify(clf_list, features_train, labels_train):

    # testing gridsearchcv
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

    new_list = []
    for clf in clf_list:
        pred = clf.predict(features_test)
        recall = recall_score(labels_test, pred)
        precision = precision_score(labels_test, pred)
        f1 = f1_score(labels_test, pred)
        new_list.append((clf, recall, precision, f1))

    return new_list



### Sort the classifier list according to their f1-score
def sort_clf(clf_list):

    clf_list_sorted = sorted(clf_list, key = lambda x:x[3], reverse=True)

    return clf_list_sorted



### Function to add PCA
def add_pca(clf_list):

    pca = PCA()
    pca_params = {'pca__n_components': [5, 10, 15]}

    delimiter = '('
    clf_list_new = []

    for classifier in clf_list:
        clf, params = classifier
        name = str(clf).split(delimiter)
        new_param = {}
        for param, values in params.iteritems():
            new_param[name[0] + '__' + param] = values

        clf_pipeline = Pipeline([('pca', pca), (name[0], clf)])
        #print clf_pipeline
        new_param.update(pca_params)
        #print new_param
        clf_list_new.append((clf_pipeline, new_param))

    return clf_list_new



### Select what features to use.

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

def add_new_features(features_list, data_dict):

    new_features = ['ratio_of_to_poi_messages',
                    'ratio_of_from_poi_messages',
                    'total_poi_messages']

    features_list = features_list + new_features

    for name in data_dict:


        if isinstance(data_dict[name]['from_this_person_to_poi'], str):
            data_dict[name]['from_this_person_to_poi'] = 0
        if isinstance(data_dict[name]['to_messages'],str):
            data_dict[name]['to_messages'] = 0
        if isinstance(data_dict[name]['from_poi_to_this_person'], str):
            data_dict[name]['from_poi_to_this_person'] = 0
        if isinstance(data_dict[name]['from_messages'], str):
            data_dict[name]['from_messages'] = 0

        try:
            ratio_of_to_poi_messages = float(data_dict[name]['from_this_person_to_poi']) /\
                                       float(data_dict[name]['to_messages'])
        except:
            ratio_of_to_poi_messages = 0
        if not isinstance(ratio_of_to_poi_messages, float):
            ratio_of_to_poi_messages = 0

        try:
            ratio_of_from_poi_messages = float(data_dict[name]['from_poi_to_this_person']) /\
                                         float(data_dict[name]['from_messages'])
        except:
            ratio_of_from_poi_messages = 0
        if not isinstance(ratio_of_from_poi_messages, str):
            ratio_of_from_poi_messages = 0

        data_dict[name]['total_poi_messages'] = data_dict[name]['from_this_person_to_poi'] +\
                                                data_dict[name]['from_poi_to_this_person']

        data_dict[name]['ratio_of_to_poi_messages'] = ratio_of_to_poi_messages
        data_dict[name]['ratio_of_from_poi_messages'] = ratio_of_from_poi_messages



    return features_list, data_dict


new_features = ['ratio_of_to_poi_messages',
                    'ratio_of_from_poi_messages',
                    'total_poi_messages']
# Start of the main body of the program

# loading the data set
data_dict = data_load_process()

# Adding new features
features_list, data_dict = add_new_features(features_list, data_dict)
#for name in data_dict:
#    for feat in new_features:
#        print data_dict[name][feat]
#        print isinstance(data_dict[name][feat], str)


### Store to my_data set for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 5: Data processing
"""
def preprocessing_pca(features_train, features_test):
    pca = PCA(n_components=5, svd_solver='randomized').fit(features_train)
    features_train_pca = pca.transform(features_train)
    features_test_pca = pca.transform(features_test)

    return features_train_pca, features_test_pca
"""

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
#features_train, features_test = preprocessing_pca(features_train, features_test)


### Applying classifier

# Creating the classifier list
clf_list = classifier_list()

# Add pca to the clf_lists
clf_list = add_pca(clf_list)
#print clf_list

# getting the classifiers with their best parameters
clf_list = classify(clf_list, features_train, labels_train)


# Evaluate the precision, recall and f1 scores  of the classifiers
clf_list = evaluate(clf_list, features_test, labels_test)
clf_list_sorted = sort_clf(clf_list)
print clf_list_sorted
print
clf =  clf_list_sorted[0][0]
print clf


"""
# One of the best performances: precision = 0.34315, recall = 0.1515

clf_best_till_now = Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=10, random_state=None,
             svd_solver='auto', tol=0.0, whiten=False)), ('RandomForestClassifier', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', n_jobs=1, oob_score=False, random_state=None,verbose=0, warm_start=False))])

"""

### Task 7: Dump your classifier, dataset, and features_list

dump_classifier_and_data(clf, my_dataset, features_list)