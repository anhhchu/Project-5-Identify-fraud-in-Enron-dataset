#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from split_data import SplitData
from features_selection import CheckAlgorithm

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV 
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit


### Import algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

### Task 1: Load the dataset, explore the dataset 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###############################################################################
### Task 2: Remove outliers
### Remove "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" as not a person
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

###############################################################################
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Create new features: fraction_to_poi and fraction_from_poi
for person in my_dataset:
    if my_dataset[person]['from_messages'] != 'NaN':
        my_dataset[person]['fraction_to_poi'] =\
        float(my_dataset[person]['from_this_person_to_poi'])/float(my_dataset[person]['from_messages'])
    else:
        my_dataset[person]['fraction_to_poi'] = 0
        
    if my_dataset[person]['to_messages'] != 'NaN':
        my_dataset[person]['fraction_from_poi'] =\
        float(my_dataset[person]['from_poi_to_this_person'])/float(my_dataset[person]['to_messages'])
    else:
        my_dataset[person]['fraction_from_poi'] = 0

###############################################################################
### Task 4: Select features & apply algorithms to test features effectiveness

all_features = ['poi',\
 'salary','bonus', 'long_term_incentive','total_payments',\
'exercised_stock_options', 'total_stock_value',\
 'fraction_to_poi', 'fraction_from_poi','shared_receipt_with_poi']
 
initial_data = featureFormat(my_dataset, all_features, sort_keys = True)
initial_labels, initial_features = targetFeatureSplit(initial_data)

###########################################
###Option 1. Manually select features (intuitively)

features_list1 = ['poi','salary','exercised_stock_options', 'bonus']
features_train1, features_test1, labels_train1, labels_test1 = SplitData(my_dataset,features_list1)

features_list2 = ['poi','salary','exercised_stock_options', 'bonus','long_term_incentive']
features_train2, features_test2, labels_train2, labels_test2 = SplitData(my_dataset,features_list2)

features_list3 = ['poi','fraction_to_poi', 'fraction_from_poi','shared_receipt_with_poi']
features_train3, features_test3, labels_train3, labels_test3 = SplitData(my_dataset,features_list3)


classifier_list = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier(),GaussianNB()]
### Test multiple classifiers on features_list1. 
#Call CheckAlgorithm function to tune the parameters of each classifiers and use best_estimator_ 
#to feed in test_classifier function

params_neigh = {"weights":('uniform', 'distance'), 'n_neighbors':[2,4,6,8,10],\
'metric':('euclidean','manhattan','chebyshev','minkowski')} 

params_dt = {"criterion":('gini', 'entropy'), 'min_samples_split':[2,3,4,5,6,7,8,9,10],\
'splitter':('best','random')}

params_rf = {"criterion":('gini', 'entropy'), 'n_estimators':[7,8,9,10,11,12,13,14,15],\
'max_features':('auto','sqrt','log2',None), 'min_samples_split':[2,3,4,5,6,7,8,9,10],\
'bootstrap':(True,False)} #'warm_start':(True,False)}

params_ada = {'n_estimators':[10,20,30,40,50,60],'algorithm':('SAMME', 'SAMME.R')}

params_NB = {}

params_list = [params_neigh,params_dt,params_rf,params_ada,params_NB]

'''i=0
while i< len(classifier_list):    
    #print CheckAlgorithm(features_list1,features_train1, labels_train1,classifier_list[i],my_dataset, params_list[i])
    #print CheckAlgorithm(features_list2,features_train2,labels_train2,classifier_list[i],my_dataset, params_list[i])
    print CheckAlgorithm(features_list3,features_train3,labels_train3,classifier_list[i],my_dataset, params_list[i])
    i+=1'''

#print test_classifier(GaussianNB(), my_dataset, features_list1)
#print test_classifier(GaussianNB(), my_dataset, features_list2)
#
### Among the above classifiers, only KNeighborsClassifier is affected by scaled features, 
# => perform MinMaxScaler and GridSearch on KNeighborsClassifier only. 

'''pipe_scale = Pipeline([("scaler", MinMaxScaler()), ("Classifier", KNeighborsClassifier())])
param_grid_neigh = dict(Classifier__n_neighbors=[2,4,6,8,10],\
                  Classifier__weights=['uniform', 'distance'],\
                  Classifier__metric=['euclidean','manhattan','chebyshev','minkowski'])
                  
grid_scale = GridSearchCV(pipe_scale, param_grid=param_grid_neigh,scoring = 'f1') 
grid_scale.fit(features_train1, labels_train1)
print test_classifier(KNeighborsClassifier(), my_dataset, features_list1)'''

#######################################

###Option 2: Select Kbest features     
         
k_list = [4,5,6,7,8,9]
'''for k in k_list:
    selected = SelectKBest(f_classif,k).fit(initial_features,initial_labels)
    best_feature_list = [all_features[n] for n in selected.get_support(indices=True)]
    print best_feature_list
    
    #print selected.pvalues_
    #print selected.pvalues_
    features_train4, features_test4, labels_train4, labels_test4 = SplitData(my_dataset,best_feature_list)
    
    i=0
    while i< len(classifier_list):    
        #print selected.scores_
        print CheckAlgorithm(best_feature_list,features_train4, labels_train4,classifier_list[i],my_dataset, params_list[i]) 
        i+=1'''
    
###############################################################################
### Task 5: Tune the most effective algorithm  
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Using the result from Option 2 in Test 4, choose the most effective algorithm
features_list = features_list1

clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='chebyshev',
           metric_params=None, n_neighbors=4, p=2, weights='distance')

###############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)