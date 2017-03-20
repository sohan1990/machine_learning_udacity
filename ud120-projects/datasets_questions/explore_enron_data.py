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
import numpy as np
from feature_format import featureFormat
from feature_format import targetFeatureSplit

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#print(len(enron_data))
#print(len(enron_data["METTS MARK"]))
#print(enron_data)

count = 0
for persons in enron_data:
    if enron_data[persons]['poi'] == True:
        count = count +1
#print(count)

count_y = 0
count_n = 0
fhand = open("../final_project/poi_names.txt")
for lines in fhand:
    lines = lines.rstrip()
    if lines.startswith('(y)'):
        count_y += 1
    elif lines.startswith('(n)'):
        count_n += 1
#print(count_y+count_n)

#print(enron_data["PRENTICE JAMES"]['total_stock_value'])

#print(enron_data["SKILLING JEFFREY K"]['total_payments'])
#print(enron_data["LAY KENNETH L"]['total_payments'])
#print(enron_data["FASTOW ANDREW S"]['total_payments'])

count_salary = 0
count_email = 0
for names in enron_data:
    if enron_data[names]['salary'] != 'NaN':
        count_salary += 1
    if enron_data[names]['email_address'] != 'NaN':
        count_email += 1

#print(count_salary)
#print(count_email)

# total no of people with no record of total payments
count = 0.0
for names in enron_data:
#    print(enron_data[names]['total_payments'])
    if enron_data[names]['total_payments'] == 'NaN' or enron_data[names]['total_payments'] == 0:
        count += 1
#print(count/len(enron_data))
#print(count)

# total no of POI with no record of total payments
count_poi = 0.0
count = 0
for names in enron_data:
    if enron_data[names]['poi'] == True:
        count_poi += 1
        if enron_data[names]['total_payments'] == 'NaN':
            count +=1
#print(count/len(enron_data))
#print(count_poi)

#print(len(enron_data))

