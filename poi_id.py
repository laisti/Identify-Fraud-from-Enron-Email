#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from time import time
from tester import dump_classifier_and_data

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### The code is moved to Task 3

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

### Check if our dataset consists of people with all NaN features
for person in data_dict:
    if data_dict[person]['to_messages'] == 'NaN' and \
        data_dict[person]['shared_receipt_with_poi'] == 'NaN' and \
        data_dict[person]['from_messages'] == 'NaN' and \
        data_dict[person]['from_this_person_to_poi'] == 'NaN' and \
        data_dict[person]['email_address'] == 'NaN' and \
        data_dict[person]['from_poi_to_this_person'] == 'NaN' and \
        data_dict[person]['salary'] == 'NaN' and \
        data_dict[person]['deferral_payments'] == 'NaN' and \
        data_dict[person]['total_payments'] == 'NaN' and \
        data_dict[person]['exercised_stock_options'] == 'NaN' and \
        data_dict[person]['bonus'] == 'NaN' and \
        data_dict[person]['restricted_stock'] == 'NaN' and \
        data_dict[person]['restricted_stock_deferred'] == 'NaN' and \
        data_dict[person]['total_stock_value'] == 'NaN' and \
        data_dict[person]['expenses'] == 'NaN' and \
        data_dict[person]['loan_advances'] == 'NaN' and \
        data_dict[person]['other'] == 'NaN' and \
        data_dict[person]['director_fees'] == 'NaN' and \
        data_dict[person]['deferred_income'] == 'NaN' and \
        data_dict[person]['long_term_incentive'] == 'NaN':
        print person

### Remove LOCKHART EUGENE E from our dataset
data_dict.pop('LOCKHART EUGENE E', 0)

### Remove TOTAL from our dataset
data_dict.pop('TOTAL', 0)

### Remove 'email_address' column
for key in data_dict:
    features_dict = data_dict[key]
    features_dict.pop('email_address', 0)

### Change missing data by setting their value to 0
for key in data_dict:
    features_dict = data_dict[key]
    for feature in features_dict:
        if features_dict[feature] == 'NaN':
            features_dict[feature] = 0

### Task 3: Create new feature(s)

def testClassifier(classifier):
    """
    Function uses classifier and prints all its metrics
    Arguments: classifier
    """
    clf = classifier
    t0 = time()
    clf = clf.fit(features_train, labels_train)
    print 'Training time: ', t0
    t1 = time()
    pred = clf.predict(features_test)
    print 'Testing time: ', t1

    ### metrics
    features_accuracy = accuracy_score(labels_test, pred)
    features_precision_score = precision_score(labels_test, pred)
    features_recall_score = recall_score(labels_test, pred)
    features_f1_score = f1_score(labels_test, pred)

    print 'Accuracy: ', features_accuracy
    print 'Precision: ', features_precision_score
    print 'Recall: ', features_recall_score
    print 'F1 score:', features_f1_score

### List of original features
features_list_original = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', \
                'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', \
                'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', \
                'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', \
                'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_original, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Split data into training and testing datasets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

### Use DecisionTreeClassifier for all original features
print 'Decision Tree Classifier for original features list:'
testClassifier(DecisionTreeClassifier(random_state=42))

def computeRatio(poi_messages, all_messages):
    """
    Function computes the ratio of messages sent and received from a POI
    Arguments: poi_messages, all_messages
    Return: ratio of arguments
    """
    if all_messages != 0:
        ratio = float(poi_messages) / float(all_messages)
    else:
        ratio = 0
    return ratio

### Create 2 new features: "ratio_to_poi" and "ratio_from_poi"
for key in my_dataset:
    features_dict = my_dataset[key]
    ratio_to_poi = computeRatio(features_dict["from_this_person_to_poi"], features_dict["from_messages"])
    features_dict["ratio_to_poi"] = ratio_to_poi
    ratio_from_poi = computeRatio(features_dict["from_poi_to_this_person"], features_dict["to_messages"])
    features_dict["ratio_from_poi"] = ratio_from_poi

### List of original features + new features
features_list_plus_new = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', \
                'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', \
                'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', \
                'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', \
                'deferred_income', 'long_term_incentive', 'from_poi_to_this_person', \
                'ratio_to_poi', 'ratio_from_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_plus_new, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Split data into training and testing datasets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

### Use DecisionTreeClassifier for all original features and new features
print 'Decision Tree Classifier for all original features and new features:'
testClassifier(DecisionTreeClassifier(random_state=42))

### List of original and new features beside 'poi'
features_list_without_poi = ['salary', 'to_messages', 'deferral_payments', 'total_payments', \
                'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', \
                'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', \
                'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', \
                'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_without_poi, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Use SelectKBest classifier for selection of 9 best features from our dataset
slc = SelectKBest(k=9)
slc = slc.fit(features, labels)
features_list = [features_list_without_poi[i] for i in slc.get_support(indices=True)]

### Insert 'poi' to the best_features list
features_list.insert(0, 'poi') 

print 'Best features using SelectKBest: ', features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Split data into training and testing datasets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
### Use DecisionTreeClassifier for best 9 features and 'poi'
print 'Decision Tree Classifier for best 9 features and "poi":'
testClassifier(DecisionTreeClassifier(random_state=42))

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Adaboost
print 'Adaboost Classifier:'
testClassifier(AdaBoostClassifier(random_state=42))

### Random Forest
print 'Random Forest Classifier:'
testClassifier(RandomForestClassifier(random_state=42))

### K-Nearest Neighbors
print 'K-Nearest Neighbors Classifier:'
testClassifier(KNeighborsClassifier())

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Random Forest using GridSearchCV
print 'Random Forest using GridSearchCV:'
param_grid = {"n_estimators": [1, 10],
              "criterion": ('gini', 'entropy'),
              "max_features": ('auto', 'sqrt', 'log2'),
              "random_state": [42]}

testClassifier(GridSearchCV(RandomForestClassifier(), param_grid))

### K-Nearest Neighbors using GridSearchCV
print 'K-Nearest Neighbors using GridSearchCV:'
param_grid = {"n_neighbors": [1, 5],
              "weights": ('uniform', 'distance'),
              "algorithm": ('auto', 'ball_tree', 'kd_tree', 'brute')}

testClassifier(GridSearchCV(KNeighborsClassifier(), param_grid))

### K-Nearest Neighbors using GridSearchCV and StandardScaler
print 'K-Nearest Neighbors using GridSearchCV and StandardScaler:'
knn = KNeighborsClassifier()
estimators = [('scale', StandardScaler()), ('knn', knn)]
pipeline = Pipeline(estimators)
parameters = {'knn__n_neighbors': [1, 5],
              'knn__weights': ('uniform', 'distance'),
              'knn__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}

testClassifier(GridSearchCV(pipeline, parameters))

### Selected algorithm

### Adaboost
print 'Adaboost Classifier:'
testClassifier(AdaBoostClassifier(random_state=42))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(AdaBoostClassifier(random_state=42), my_dataset, features_list)
