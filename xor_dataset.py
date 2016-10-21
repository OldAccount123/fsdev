#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:49:33 2016

@author: Matej Gazda

The tenth (or in python indexing 9 ) and 20th (or in python indexing 19) are 
relevant features. Other is just random. The class label is given by X = f10 xor f20

Minimum number of features is 20, but can by easily changed rewriting the code.
"""

import numpy as np

#N_SAMPLES = 50 
#N_FEATURES = 10
def xor_dataset(N_SAMPLES, N_FEATURES):
    
    if N_FEATURES < 20:
        raise ValueError("Minumum number of features is 20. Increase number of features in the input argument")
        
    relevant_feat1 = np.random.randint(2, size=(N_SAMPLES,1))
    relevant_feat2 = np.random.randint(2, size=(N_SAMPLES,1))
    
    # create class label and convert bool array to int array
    class_label = np.logical_xor(relevant_feat1, relevant_feat2).astype(int)
    
    # create noise features 
    noise_feat1 = np.random.randint(2, size=(N_SAMPLES, 9))
    noise_feat2 = np.random.randint(2, size=(N_SAMPLES, N_FEATURES-9-9-2))
    noise_feat3 = np.random.randint(2, size=(N_SAMPLES, 9))
        
    # stack features noise , relevant, noise relevant, noise
    features = np.hstack((noise_feat1, relevant_feat1, noise_feat3, relevant_feat2, noise_feat2))
    
    return class_label, features
    