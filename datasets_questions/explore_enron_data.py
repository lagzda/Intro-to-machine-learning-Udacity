#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Total entries:", len(enron_data)
print "Pois:", len([enron_data[key] for key in enron_data.keys() if enron_data[key]["poi"]==1])
print "Quantified salary:", len([enron_data[key] for key in enron_data.keys() if enron_data[key]["salary"] != "NaN"])
print "Emails:", len([enron_data[key] for key in enron_data.keys() if enron_data[key]["email_address"] != "NaN"])
print "Missing payments:", len([enron_data[key] for key in enron_data.keys() if enron_data[key]["total_payments"] == "NaN"])
print "Percentage of missing payments:",len([enron_data[key] for key in enron_data.keys() if enron_data[key]["total_payments"] == "NaN"]) / float(len(enron_data))
print "Missing POIs payments:", len([enron_data[key] for key in enron_data.keys() if enron_data[key]["total_payments"] == "NaN" and enron_data[key]["poi"]==1])






