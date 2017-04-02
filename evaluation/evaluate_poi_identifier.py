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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here 
clf = DecisionTreeClassifier()
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

p_score = precision_score(labels_test, pred)
r_score = recall_score(labels_test, pred)

print "No of predicted POIs:", len(np.where(pred == 1.)[0])
print "No of true positives:", len([x for x in range(len(pred)) if pred[x] == 1. and pred[x] == labels_test[x]])
print "No of people in test set:", len(pred)
print "Precission and recall scores:", p_score, r_score

