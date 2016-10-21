# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.datasets import make_classification


def generate_simple_dataset(N_samples, N_features, N_relevant_features):
    """
    Function generates very simple two-class database.
    
    Two class database consists of N_samples samples and N_features. From these
    N_relevant features are relevant to the class label. Relevant features are 
    generated as random integers within interval [0,4] for class 0, and from 
    interval [100, 120] for class 1. Noisy features, not relevant for classification,
    are generated randomly within interval [0, 150]. This is extremly simple 
    machine learning task intended only for first level testing of feature selection
    methods and classifiers.
    
    Arguments:
    N_samples
    N_features
    N_relevant_features
    
    features -  dataset features N_samples x N_features
    lebels   - class labels 
    relevant_indices - column index in features array that contains relevant features    
     
    """
    
    relevant_features_class0 = np.random.randint(0,4,(N_samples/2,N_relevant_features))
    label_class0 = np.zeros((N_samples/2,1))
    relevant_features_class1 = np.random.randint(100,120,(N_samples/2,N_relevant_features))
    label_class1 = np.ones((N_samples/2,1))
    noisy_features = np.random.randint(0,150, (N_samples,N_features-N_relevant_features))
    noisy_features_marked = np.vstack((np.zeros((1,N_features-N_relevant_features)), noisy_features))
    label = np.vstack((label_class0, label_class1))
    
    relevant_features = np.vstack((relevant_features_class0, relevant_features_class1))
    relevant_features_marked = np.vstack((np.ones((1,N_relevant_features)), relevant_features)) # marking position opf relevant features
    
    all_features_marked = np.hstack((relevant_features_marked, noisy_features_marked))
    
    #shufle columns
    features_marked = np.transpose(all_features_marked)
    np.random.shuffle(features_marked)
    features_marked = np.transpose(features_marked)
    
    features=features_marked[1:,:]
    marks = features_marked[0,:]
    
    relevant_indices = np.where(marks==1)[0]
    #shufle rows
    dataset = np.hstack((label, features))
    np.random.shuffle(dataset)
    
    label = dataset[:,0]
    features = dataset[:,1:]
    
    return features, label, relevant_indices
    
    
    
def generate_clasification(N_samples, N_features, N_relevant_features):
    """
    Function generates very simple two-class database.
    
    Two class database consists of N_samples samples and N_features. From these
    N_relevant features are relevant to the class label. Relevant features are 
    generated as random integers within interval [0,4] for class 0, and from 
    interval [100, 120] for class 1. Noisy features, not relevant for classification,
    are generated randomly within interval [0, 150]. This is extremly simple 
    machine learning task intended only for first level testing of feature selection
    methods and classifiers.
    
    Arguments:
    N_samples
    N_features
    N_relevant_features
    
    features -  dataset features N_samples x N_features
    lebels   - class labels 
    relevant_indices - column index in features array that contains relevant features    
     
    """
    
    NOF_SAMPLES = N_samples
    NOF_RELEVANT_FEATURES = N_relevant_features
    NOF_RANDOM_FEATURES = N_features- N_relevant_features
    
    X_relevant, y = make_classification(n_samples=NOF_SAMPLES, n_features=NOF_RELEVANT_FEATURES, n_informative=NOF_RELEVANT_FEATURES,
                               n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=5, random_state=0, class_sep=5)
    
    X_scaled = (X_relevant - X_relevant.min())/(X_relevant.max()-X_relevant.min())
    
    X_random = np.random.rand(NOF_SAMPLES, NOF_RANDOM_FEATURES)
    
    noisy_features = X_random
    noisy_features_marked = np.vstack((np.zeros((1,NOF_RANDOM_FEATURES)), noisy_features))
    
    relevant_features = X_scaled
    relevant_features_marked = np.vstack((np.ones((1,NOF_RELEVANT_FEATURES)), relevant_features)) # marking position opf relevant features
    
    all_features_marked = np.hstack((relevant_features_marked, noisy_features_marked))
    
    #shufle columns
    features_marked = np.transpose(all_features_marked)
    np.random.shuffle(features_marked)
    features_marked = np.transpose(features_marked)
    
    features=features_marked[1:,:]
    marks = features_marked[0,:]
    
    relevant_indices = np.where(marks==1)[0]
    
    return features, y, relevant_indices