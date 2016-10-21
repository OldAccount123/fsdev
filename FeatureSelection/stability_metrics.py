# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:01:34 2015

Metrics to evaluate stability of feature selection.


ARGUMENTS:

indices_selected    -  indices of selected selected in bootsrtaping repetitions
                       nof repetitions x nof_selected features
nof_features        -  number of all features in dataset
return              -  selected stability index   


Notes:
Implemented stability measures:
stability index [1]
average_tanimoto_index [3,5:(eq4,eq18)]
weightened_consistency [5]


stability_index: kuncheva index 
([1]: Kuncheva, A STABILITY INDEX FOR FEATURE SELECTION)
dunne hamming distance 
([2]:Dunne, Solutions to Instability Problems with Sequential
                                       Wrapper-based Approaches to Feature
                                       Selection)
                           - also defined in [1]
tanimoto_index: Tanimoto distance/Kalousis 
 ([3]: Kalousis: Stability of feature selection algorithms: a
                                           study on high-dimensional spaces
                           -also defined in [1]
 Moulos (modification of dunne,without normalisation by number of features) 
([4]:Moulos:Stability of Feature Selection Algorithms for Classification in High-
            Throughput Genomics Datasets

[5] Somol,Novovicova:Evaluating Stability and Comparing Output of Feature Selectors that
Optimize Feature Subset Cardinality Petr Somol and Jana Novovicova 

"""

from __future__ import division
import numpy as np


def stability_index(indices_selected, nof_features):
    
    nof_reps, nof_selected_features = indices_selected.shape
    kuncheva_index = 0
    
    for ii_row in range(nof_reps - 1):
        for jj_row in range(ii_row+1, nof_reps):
            r = np.intersect1d(indices_selected[ii_row,:], indices_selected[jj_row,:]).size
            consistency_index = ( r*nof_features - pow(nof_selected_features, 2) ) / ( nof_selected_features*(nof_features - nof_selected_features) ) #according to equation in [1]
            kuncheva_index = consistency_index + kuncheva_index
            
    kuncheva_index_norm = (2/(nof_reps*(nof_reps-1)))*kuncheva_index; #[1]eq(6)
    return kuncheva_index_norm

def average_tanimoto_index(indices_selected, nof_features):
    
    nof_reps, nof_selected_features = indices_selected.shape
    kalousis_index = 0
    
    for ii_row in range(nof_reps - 1):
        for jj_row in range(ii_row+1, nof_reps):
            kalousis_pair = np.intersect1d( indices_selected[ii_row,:], indices_selected[jj_row,:] ).size / np.union1d( indices_selected[ii_row,:], indices_selected[jj_row,:] ).size #according to equation in [1]
            kalousis_index = kalousis_pair + kalousis_index
            
    kalousis_index_norm = (2/(nof_reps*(nof_reps-1)))*kalousis_index; # normovanie [5]eq(18)=average tanimoto index
    return kalousis_index_norm
                
def average_normalized_hamming_index(indices_selected, nof_features):
    nof_reps, nof_selected_features = indices_selected.shape
    nhi_index = 0
    
    for ii_row in range(nof_reps - 1):
        for jj_row in range(ii_row+1, nof_reps):    
            nhi = 1 -  (np.setdiff1d( indices_selected[ii_row,:], indices_selected[jj_row,:] ).size + np.setdiff1d( indices_selected[jj_row,:], indices_selected[ii_row,:] ).size)/nof_features        
            nhi_index = nhi + nhi_index 
    
    anhi = (2/(nof_reps*(nof_reps-1)))*nhi_index;
    return anhi  

def weightened_consistency(indices_selected, nof_features):
    
    nof_reps, nof_selected_features = indices_selected.shape
    rows_indices_selected, cols_indices_selected = indices_selected.shape # rows correspond to n in somol
    N = nof_reps * nof_selected_features # corresponds to N in somol

    hist_data, bin_edges = np.histogram(indices_selected, bins=nof_features,density=False) # pocetnost pre kazdu feature

    cw_temp = 0 
    cw = 0  # corresponds to CW(S) in homol
    for feature in range(nof_features):
        F_f = hist_data[feature]
        cw_temp = (F_f/N)*( (F_f-1)/(nof_reps-1) )
        cw = cw + cw_temp
        
    # cwrel dle Somol (14)-(17)
    D = N % nof_features
    H = N % nof_reps
 
    cw_min = ( pow(N,2) - nof_features*(N-D) - pow(D,2) )/( nof_features*N*(nof_reps-1) )
    cw_max = ( pow(H,2) + N*(nof_reps-1) - H*nof_reps ) / ( N*(nof_reps-1) )
    cw_rel = (cw - cw_min)/(cw_max-cw_min)
   
    return cw, cw_rel    
    
    
    
    
    
    
    



































