# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 19:36:47 2016

@author: Matej Gazda
"""

from IPython import get_ipython

# get_ipython().magic('reset -sf')  # clean workspace if the script is run in the same console
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score, train_test_split
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from FeatureSelection import fs_filter
from ballotbox.ballot import BallotBox
from ballotbox.singlewinner.preferential import BordaVoting
from Voting import voting
from Voting.utils import transf_ranking, shuffle_features
import pandas as pd
from xor_dataset import xor_dataset

# DATASET TO USE. Corrall, Madelon, XOR
DATASET = ['Corrall']
FS_METHODS_NAMES = ['Fisher', 'Trace_Ration', 'LL21', 'RFS', 'BDIST', 'TTEST', 'ANOVA', 'PEARSON', 'RELIEFF', 'MIC',
                    'GINI']
CHOSEN_FS_METHODS = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
VOTING_METHODS_NAMES = ['borda', 'STV', 'kemeny_young', 'plurality', 'own_borda', 'borda_weighted']
CHOSEN_VOTING_METHODS = [0, 1, 3, 4, 5]
AGGR_METHODS_NAMES = ['min', 'max', 'mean', 'median']
CHOSEN_AGGR_METHODS = [0, 1, 2, 3]
N_OF_FS = len(CHOSEN_FS_METHODS)  # number of feature-selection methods.
N_OF_LOOPS = 1

# fill the columns according to methods used.
columns = []
for i in CHOSEN_FS_METHODS:
    columns.append(FS_METHODS_NAMES[i])
for i in CHOSEN_VOTING_METHODS:
    columns.append(VOTING_METHODS_NAMES[i])
for i in CHOSEN_AGGR_METHODS:
    columns.append(AGGR_METHODS_NAMES[i])

# CREATE PANDAS DATA FRAME AND SAVE THERE RESULTS.
results = pd.DataFrame(index=np.arange(N_OF_LOOPS), columns=columns)

# load CSV file, transform it to 2d np array.
if DATASET[0] == 'Corrall':
    N_feats = 200
    X_ = pd.read_csv('files/data3.csv').as_matrix(columns=None)
    y_ = np.hstack(pd.read_csv('files/data2.csv').as_matrix(columns=None)).astype(int)
    true_feats = [0, 1, 2, 3, 4]

    X = X_[:, :N_feats]
    y = y_[:N_feats]

    X_shuffled, true_feats_shuffle = shuffle_features(X, true_feats)
elif DATASET[0] == 'LED':
    data = np.loadtxt("files/LED25_n0.txt", delimiter=",")
    X = data[:, :(data.shape[1] - 1)]
    y = np.ravel(data[:, -1:])
    true_feats = np.arange(7)
    X_shuffled, true_feats_shuffle = shuffle_features(X, true_feats)

elif DATASET[0] == 'Madelon':
    N_inf = 5
    X, y = make_classification(n_features=500, n_redundant=5, n_informative=N_inf,
                               random_state=6, n_clusters_per_class=16,
                               n_classes=2, flip_y=0.01, n_repeated=0,
                               class_sep=2, shuffle=False
                               )
    true_feats = np.arange(N_inf)
    X_shuffled, true_feats_shuffle = shuffle_features(X, true_feats)

elif DATASET[0] == 'XOR':
    y, X = xor_dataset(500, 500)
    y = np.hstack(y)
    true_feats = [9, 19]

    # true_feats_shuffle = true_feats
    # X_shuffled = X
    X_shuffled, true_feats_shuffle = shuffle_features(X, true_feats)

for loops_count in np.arange(N_OF_LOOPS):
    # X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y,
    #                                                    test_size=0.5,
    #                                                    random_state=0)
    X_train = X_shuffled
    y_train = y

    # X_train = X_train[:, :450]
    # y_train = y_train[:450]

    # NUMBER OF FEATURES
    N_FEATURES = np.shape(X_train)[1]
    # NUMBER OF SAMPLES OF DATA.
    NUMBER_OF_SAMPLES = np.shape(X_train)[0]

    # save all the available fs functions to a list.
    fs_methods = [fs_filter.fs_fischer,
                  fs_filter.fs_trace_ration,
                  fs_filter.fs_ll21,
                  fs_filter.fs_RFS,
                  fs_filter.fs_ttest,
                  fs_filter.fs_bdist,
                  fs_filter.fs_anova,
                  fs_filter.fs_pearson,
                  fs_filter.fs_relief,
                  fs_filter.fs_mic,
                  fs_filter.fs_gini,
                  ]

    # REF RELIEF
    temp_rel = fs_methods[8](X_train, y_train)
    # ==============================================================================
    # CREATE 1D NESTED LIST WITH FOLLOWING STRUCTURE:
    # FEATURES_SELECTED[0]... FEATURES_SELECTED[N], N = NUMBER OF FS METHODS USED
    # FEATURES_SELECTED[N],... CONTAINS SCORES OF FEATURES ACCORDING TO NTH METHOD
    #                          SAVED IN NP.ARRAY
    # ==================================================================================
    # basic feature selection methods
    features_selected = list()
    for i in CHOSEN_FS_METHODS:
        if fs_methods[i].__name__ == 'fs_trace_ration':
            temp_results = fs_filter.fs_trace_ration(X_train, y_train, N_FEATURES, style='fisher')
        elif fs_methods[i].__name__ == 'fs_RFS':
            temp_results = fs_filter.fs_RFS(X_train, y_train, gamma=0.1)
        elif fs_methods[i].__name__ == 'fs_ll21':
            temp_results = fs_filter.fs_ll21(X_train, y_train, gamma=0.1)
        else:
            temp_results = fs_methods[i](X_train, y_train)
        features_selected.append(temp_results.astype(int))
        results[FS_METHODS_NAMES[i]][loops_count] = temp_results

    # transform features_selected into 2d np array.the first is the most important feature, etc
    features_ranking = np.zeros([N_OF_FS, N_FEATURES])
    for i in np.arange(len(features_ranking)):
        features_ranking[i] = features_selected[i]

    print 'FINISHED FS METHODS, STARTED VOTING METHODS'
    # here we store all the voting methods into a list.
    voting_methods = [voting.voting_borda,
                      voting.voting_STV,
                      voting.voting_kemeny_young,
                      voting.voting_plurality,
                      voting.voting_own_borda,
                      voting.voting_borda_weighted,
                      ]

    print 'STARTING THE VOTING METHODS'
    voting_results = []
    for i in CHOSEN_VOTING_METHODS:
        if voting_methods[i].__name__ == 'voting_STV':
            temp_results = list(voting_methods[i](features_ranking, 5)['winners'])
        elif voting_methods[i].__name__ == 'voting_borda_weighted':
            temp_results = voting_methods[i](features_ranking, 5, mode='step', N=100)
        else:
            temp_results = voting_methods[i](features_ranking, 5)
        voting_results.append(temp_results)
        results[VOTING_METHODS_NAMES[i]][loops_count] = temp_results
    # transform the array into: nth index = nth feature. nth value = feature rank.
    features_ranking_transf = transf_ranking(features_ranking)
    # here we store all the aggr methods into a list.
    aggr_results = []
    aggr_methods = [voting.voting_min,
                    voting.voting_max,
                    voting.voting_mean,
                    voting.voting_median]
    for i in CHOSEN_AGGR_METHODS:
        temp_results = aggr_methods[i](features_ranking_transf, 5)
        aggr_results.append(temp_results)
        results[AGGR_METHODS_NAMES[i]][loops_count] = temp_results

    print "RESULTS"
    print results


# print results
