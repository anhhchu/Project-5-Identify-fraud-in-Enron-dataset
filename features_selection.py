# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:37:18 2015

@author: jasmin may
"""
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from split_data import SplitData

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV 

from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier


def CheckAlgorithm(features_list,features_train, labels_train,classifier,my_dataset, params):
    
    grid = GridSearchCV(classifier, param_grid=params,scoring = 'f1')
    grid.fit(features_train, labels_train)
    return test_classifier(grid.best_estimator_,my_dataset,features_list)



   
    