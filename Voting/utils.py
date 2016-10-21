import numpy as np
from itertools import combinations
from collections import Counter
from scipy.stats.mstats import mode


def shuffle_features(X,true_feats):
    feat_label = np.arange(X.shape[1]) # generate feature labels. equal to number of feats
    X_labeled = np.vstack((feat_label,X)) # insert zero row being lael

    
    np.random.shuffle(np.transpose(X_labeled))  # shuffle columns=features
    
    # find new position of true features
    true_feats_shuffle = []   
    for item in true_feats:
        temp=np.where(X_labeled[0]==item)
        true_feats_shuffle.append(temp[0])
    X_shuffled=X_labeled[1:]

    return X_shuffled, np.array(true_feats_shuffle)
    
    
    
    
def _build_graph(ranks):
    n_voters, n_candidates = ranks.shape
    edge_weights = np.zeros((n_candidates, n_candidates))
    for i, j in combinations(range(n_candidates), 2):
        preference = ranks[:, i] - ranks[:, j]
        h_ij = np.sum(preference < 0)  # prefers i to j
        h_ji = np.sum(preference > 0)  # prefers j to i
        if h_ij > h_ji:
            edge_weights[i, j] = h_ij - h_ji
        elif h_ij < h_ji:
            edge_weights[j, i] = h_ji - h_ij
    return edge_weights


def make_list(features_ranking):
    """
    Parameters
    ----------
    features_ranking

    Returns
    -------
    list of dicts, input format for ballotbox lib.
    """
    fs_ranks_list = []
    for i in np.arange(len(features_ranking)):
        fs_ranks_dict = {int(features_ranking[i][n]): n for n in np.arange(len(features_ranking[0]))}
        fs_ranks_list.append(fs_ranks_dict)
    return fs_ranks_list


def get_most_common(arr):
    """
    finds the most common number in 1d np array.
    Parameters
    ----------
    arr: 1d np array.

    Returns
    -------
    returns the most common number in a 1d np array.
    """
    return mode(arr)[0][0]


def transf_ranking(features_ranking):
    """
    transforms the featuresa ranking 2d array according to this rule:
    1st col = places of first candidate according to voters.
    2nd col = places of second candidate according to voters.
    Parameters
    ----------
    features_ranking
    2d np array.
    each row represents one voter.
    Returns
    -------
    returns the transformed value.
    """
    features_ranking_transf = np.zeros((len(features_ranking), len(features_ranking[0])))
    for feature in np.arange(len(features_ranking[0])):
        n_method = 0
        for method in np.arange(len(features_ranking)):
            features_ranking_transf[n_method, features_ranking[method][feature]] = feature
            n_method += 1
    return features_ranking_transf


def get_index(arr, item):
    """
    gets index of a unique item in an np.array
    Parameters
    ----------
    arr - array to be searched
    item - find index of the given item in arr
    Returns
    -------
    returns the index of the item as an integer.
    """
    index = np.where(arr == item)
    return index[0][0]


def get_coeff(position, mode='step', N=10):
    """
    gets the weight of feature for weighted borda voting method
    Parameters
    ----------
    position: integer, position of the feature
    mode: the mode how is the coeff calculated
    N: optional parameter, used for all of the modes
    Returns
    -------
    score of the feature after weighting (integer)
    """
    if mode == 'step':
        if position > N:
            return 0
        else:
            return 1
    if mode == 'twostep':
        return 0
    if mode == 'exp':
        return 0
