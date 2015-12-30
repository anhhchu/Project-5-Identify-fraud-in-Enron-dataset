#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Load the dataset, dataset information

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### prepare the features and labels
print "number of data point (people) in the dataset:", len(data_dict)
print "number of features in the dataset:", len(data_dict["LAY KENNETH L"])

#count the number of person of interest
n=0
for person in data_dict:
    if data_dict[person]["poi"] == True:
        n=n+1
print "Number of POIs:", n

### check for missing values
# define an empty dictionary to store number of missing values


###############################################################################
### Task 2: Remove outliers
data_dict.pop("TOTAL")

###############################################################################
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

print "number of data points after removing TOTAL", len(my_dataset)

### Create new features: fraction_to_poi and fraction_from_poi
for person in my_dataset:
    if my_dataset[person]['from_messages'] != 'NaN':
        my_dataset[person]['fraction_to_poi'] =\
        float(my_dataset[person]['from_this_person_to_poi'])/float(my_dataset[person]['from_messages'])
    else:
        my_dataset[person]['fraction_to_poi'] = 0

for person in my_dataset:
    if my_dataset[person]['to_messages'] != 'NaN':
        my_dataset[person]['fraction_from_poi'] =\
        float(my_dataset[person]['from_poi_to_this_person'])/float(my_dataset[person]['to_messages'])
    else:
        my_dataset[person]['fraction_from_poi'] = 0
 
### Scale financial features

###############################################################################
### Task 4: Select features 
### features_list is a list of strings, each of which is a feature name, including the new features
### The first feature must be "poi".

all_features = ['poi',\
 'salary','bonus', 'long_term_incentive','total_payments',\
 'exercised_stock_options', 'total_stock_value',\
 'fraction_to_poi', 'fraction_from_poi','shared_receipt_with_poi'] # You will need to use more features

initial_data = featureFormat(my_dataset, all_features, sort_keys = True)
initial_labels, initial_features = targetFeatureSplit(initial_data)
print initial_data[0]

### Draw scatterplot to inspect features in all_features 
import matplotlib.pyplot
def draw_scatterplot(data, index, features_list):
    for point in data:
        features_list[index] = point[index]
        features_list[index+1] = point[index+1]
        matplotlib.pyplot.scatter(features_list[index], features_list[index+1], color="blue")
    matplotlib.pyplot.show() 

print "Scatterplot:"
print "salary and bonus"
draw_scatterplot(initial_data, 1, all_features)

print "long term incentive and total payment"
draw_scatterplot(initial_data, 3, all_features) 

print "exercised_stock_options and total_stock_value"
draw_scatterplot(initial_data, 5, all_features) 

print "fraction of emails from_this_person_to_poi and from_poi_to_this_person"
draw_scatterplot(initial_data, 7, all_features)