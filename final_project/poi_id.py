#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)


### Task 1: Select what features you'll use.
print "\n","="*20
print "Task 1: Select features"
print "="*20
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Feature selection comments ###
### Firstly by identifying the features which had the most NaN values I was able to remove 
### 4 features that had way over half of the missing values. After that the most logical thing
### that seemed to me was to initially use the remaining features and scale / reduce / combine
### them in later steps.
features_list = [ 'poi','salary', 'to_messages', 'total_payments', 'exercised_stock_options', 'bonus',\
'restricted_stock', 'shared_receipt_with_poi', 'total_stock_value', 'expenses', 'from_messages', 'other',\
'from_this_person_to_poi', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

### I found that using pandas to analyse and transform the data was more syntactically 
### easier so I temporarily use a DataFrame which is later turned back into an object.
pd_data = pd.DataFrame.from_dict(my_dataset, orient='index').replace("NaN", np.nan)

### Initial analysis of the data
print "No of entries:", pd_data.shape[0]
print "No of POIs:", len(pd_data[pd_data["poi"] == 1])
print "Percentage of POIs:", len(pd_data[pd_data["poi"] == 1]) / float(pd_data.shape[0]) * 100,"%"
print "Initial no of features:", pd_data.shape[1]
### Commented so it does not mess up the terminal too much
# print "Features:", pd_data.columns
# print "10 Max NaN values:\n", pd_data.isnull().sum().nlargest(n=10)
# print "10 Max NaN values in POIs: \n", pd_data[pd_data["poi"]==1].isnull().sum().nlargest(n=10)

### Task 2: Remove outliers
print "\n","="*20
print "Task 2: Remove outliers"
print "="*20
### Delete features that won't be useful (biggest NaN occurences)
pd_data.drop( pd_data.isnull().sum().nlargest(n=4).index, axis=1, inplace=True )
pd_data.drop(["email_address"], axis=1, inplace=True)
print "Potential outlier:", pd_data[pd_data["salary"] == pd_data["salary"].max()].index
# print "Potential outlier:", pd_data[pd_data["from_poi_to_this_person"] > 500]
# print "Potential outlier:", pd_data[pd_data["from_this_person_to_poi"] > 500]
# print "Potential outlier:", pd_data[pd_data["other"] > 10000000]
### Delete outliers
pd_data.drop("TOTAL", inplace=True)


### Visualise relationships between data and manually look for outliers (helper function)
def visualise_scatter(f1, half=1):
	for idx, p in enumerate(range(241,249)):
		if half == 2: idx += 8 
		plt.subplot(p)
		f2 = pd_data.columns[idx]
		f1_f2 = pd_data[[f1, f2]][pd_data["poi"] == 0]
		f1_f2_poi = pd_data[[f1, f2]][pd_data["poi"] == 1]
		f1_f2_not_na = pd_data[[f1, f2]].dropna()
		plt.scatter(f1_f2[f1], f1_f2[f2], color="blue")
		plt.scatter(f1_f2_poi[f1], f1_f2_poi[f2], color="red")
		plt.xlabel(f1)
		plt.ylabel(f2)
	plt.show()
### This was plotted in 2 halves of 8 to see all features (Uncomment bottom lines to see and change the first variable)
# visualise_scatter("from_poi_to_this_person", 1)
# visualise_scatter("from_poi_to_this_person", 2)

### Task 3: Create new feature(s)
print "\n","="*20
print "Task 3: Feature Engineering"
print "="*20
### By looking at the email features it makes more sense to engineer two new features
### that would show the percentage (or fraction) of POI emails from total sent or received emails.
### The reason for this choice is that it will more precisely show a persons contact 
### to a POI in relation to others.
pd_data["perc_poi_from"] = pd_data["from_poi_to_this_person"] / pd_data["to_messages"]
pd_data["perc_poi_to"] = pd_data["from_this_person_to_poi"] / pd_data["from_messages"]
### Then I checked how many NaN values were there (quite a lot - 59)
print "New feature 1 NaN values:", pd_data["perc_poi_from"].isnull().sum()
print "New feature 2 NaN values:", pd_data["perc_poi_to"].isnull().sum()


### Store to my_dataset for easy export below.

### Change back from pandas dataframe to a dict for compatibility
my_dataset = pd_data.replace(np.nan, "NaN").to_dict(orient="index")

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# labels, features = (np.array(labels), np.array(features))

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.preprocessing import StandardScaler
### Comparison between three feature selection tools
# k_best = SelectKBest(f_classif).fit(features, labels)
# print "Feature scores KBest:", k_best.scores_
# clf = DecisionTreeClassifier()
# clf.fit(features, labels)
# print "Feature importances DT:", clf.feature_importances_
# randomized_logistic = RandomizedLogisticRegression().fit(features, labels)
# print "Features scores RLR", randomized_logistic.scores_
# print sorted(features_list)


### Task 4: Try a variety of classifiers
print "\n","="*20
print "Task 4: Classifier"
print "="*20
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


clf = Pipeline([
  ('standardscaler', StandardScaler()),
  ('pca', PCA(n_components=10)),
  ('feature_selection', SelectKBest(f_classif, k=5)),
  ('classification', GaussianNB(priors=[0.99, 0.01]))
])
clf2 = Pipeline([
  # ('standardscaler', StandardScaler()),
  # ('pca', PCA(n_components=10)),
  ('feature_selection', SelectKBest(f_classif, k=5)),
  ('classification', RandomForestClassifier(random_state=0))
])
clf3 = Pipeline([
  # ('standardscaler', StandardScaler()),
  # ('pca', PCA(n_components=10)),
  ('feature_selection', SelectKBest(f_classif, k=5)),
  ('classification', AdaBoostClassifier(n_estimators=50, learning_rate=0.1, algorithm="SAMME", random_state=0))
])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### K Fold Cross validation
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=3, shuffle=True)
# kf.get_n_splits(features)
# features = np.array(features)
# labels = np.array(labels)
# for train_index, test_index in kf.split(features):
# 	features_train, features_test = features[train_index], features[test_index]
# 	labels_train, labels_test = labels[train_index], labels[test_index]
# 	clf.fit(features_train, labels_train)
# 	print "Training Gaussian NB score:", clf.score(features_train, labels_train)
# 	print "Testing Gaussian NB score:", clf.score(features_test, labels_test)
# 	print "Gaussian NB Precision:", precision_score(labels_test, clf.predict(features_test)), "Recall:", recall_score(labels_test, clf.predict(features_test)), "F1 Score:", f1_score(labels_test, clf.predict(features_test))
# 	print "Gaussian NB Confusion Matrix:", confusion_matrix(labels_test, clf.predict(features_test))
### End of K Fold cross validation

### Transform train and test features to only use the best 5 features
### Only uncomment when doing GridCV as these are already in the pipeline
# k_best = SelectKBest(f_classif, k=5)
# k_best.fit(features, labels)
# features_train = k_best.transform(features_train)
# features_test = k_best.transform(features_test)
### End of feature tranform

### Grid CVs (Only uncomment one at a time _ the small piece of the code above for preprocessing)

### Grid CV for Random Forest
# parameters = {'n_estimators':[10, 50, 100, 200], "criterion": ("gini", "entropy"), "max_depth": [None, 4, 10, 50, 100], "min_samples_split": [2, 50, 100], "random_state": [0]}
# forest = RandomForestClassifier()
# clf = GridSearchCV(forest, parameters)
# clf.fit(features_train, labels_train)
# print "Grid CV Random forest results:"
# print "Best score:", clf.best_score_
# print "Best params", clf.best_params_
# print "Precision:", precision_score(labels_test, clf.predict(features_test)), "Recall:", recall_score(labels_test, clf.predict(features_test))
### End of Grid CV for Random Forest 

### Grid CV for Naive Bayes
# parameters = {"priors": [[ 0.875, 0.125], [0.9, 0.1], [0.95, 0.05]]}
# nb = GaussianNB()
# clf = GridSearchCV(nb, parameters)
# clf.fit(features_train, labels_train)
# print "Grid CV Naive Bayes results:"
# print "Results: \n", pd.DataFrame(clf.cv_results_)
# print "Best score:", clf.best_score_
# print "Best params", clf.best_params_
# print "Precision:", precision_score(labels_test, clf.predict(features_test)), "Recall:", recall_score(labels_test, clf.predict(features_test))
### End of Grid CV for Naive Bayes

### Grid CV for AdaBoost
# parameters = {'n_estimators':[50, 100, 200], "learning_rate": [.1, 1.,10.,100.], "algorithm": ("SAMME", "SAMME.R"), "random_state": [0]}
# ada = AdaBoostClassifier()
# clf = GridSearchCV(ada, parameters)
# clf.fit(features_train, labels_train)
# print "Grid CV AdaBoost results:"
# print "Results: \n", pd.DataFrame(clf.cv_results_)
# print "Best score:", clf.best_score_
# print "Best params", clf.best_params_
# print "Precision:", precision_score(labels_test, clf.predict(features_test)), "Recall:", recall_score(labels_test, clf.predict(features_test))
### End of Grid CV for Adaboost

### End of Grid CVs

### Fitting of the actual algorithms (Uncommented is the final algorithm)
clf.fit(features_train, labels_train)
# clf2.fit(features_train, labels_train)
# clf3.fit(features_train, labels_train)

print "Training Gaussian NB score:", clf.score(features_train, labels_train)
# print "Training Random Forest score:", clf2.score(features_train, labels_train)
# print "Training AdaBoost score:", clf3.score(features_train, labels_train)
# # print "Test labels:", labels_test
print "Testing Gaussian NB score:", clf.score(features_test, labels_test)
# print "Testing Random Forest score:", clf2.score(features_test, labels_test)
# print "Testing AdaBoost score:", clf3.score(features_test, labels_test)

print "Gaussian NB Precision:", precision_score(labels_test, clf.predict(features_test)), "Recall:", recall_score(labels_test, clf.predict(features_test)), "F1 Score:", f1_score(labels_test, clf.predict(features_test))
# print "Random Forest Precision:", precision_score(labels_test, clf2.predict(features_test)), "Recall:", recall_score(labels_test, clf2.predict(features_test)), "F1 Score:", f1_score(labels_test, clf2.predict(features_test))
# print "AdaBoost Precision:", precision_score(labels_test, clf3.predict(features_test)), "Recall:", recall_score(labels_test, clf3.predict(features_test)), "F1 Score:", f1_score(labels_test, clf3.predict(features_test))

print "Gaussian NB Confusion Matrix:", confusion_matrix(labels_test, clf.predict(features_test))
### End of fitting of actual algorithms

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)