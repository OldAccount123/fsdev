# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 20:05:46 2016

@author: Matej Gazda
"""

import numpy as np
from ballotbox.ballot import BallotBox
from ballotbox.singlewinner.preferential import BordaVoting
from ballotbox.singlewinner.preferential.condorcet import CopelandVoting
from ballotbox.singlewinner.preferential.condorcet import KemenyYoungVoting
from ballotbox.singlewinner.plurality import FirstPastPostVoting
from pyvotecore.stv import STV
from pyvotecore.plurality import Plurality
from utils import _build_graph, make_list, transf_ranking, get_most_common, get_index, get_coeff
from itertools import permutations, combinations
from lp_solve import lp_solve


def voting_borda(features_ranking, req_winners=5):
    """
    Borda-voting method 
    
    Initially it transform list of votes into an dictionary and then performs
    borda-voting method. Returns the winner. This function repeats recursively.
    
    Parameters
    ----------
    features_ranking: 2d np array, each column contains 1 sorted vote. 
    each col = one voter.
    req_winners: integer, how many winners to return.
    """
    fs_ranks_list = make_list(features_ranking)
    fs_winners = []
    # borda count
    for i in np.arange(req_winners):
        bb = BallotBox(method=BordaVoting, mode="standard")
        # add votes from every used feature selection method    
        for j in np.arange(len(features_ranking)):
            bb.add_votes(fs_ranks_list[j], 1)
        vote_winner = int(bb.get_winner()[0][1])  # get winner
        fs_winners.append(vote_winner)  # save it to fs_winners list
        # delete the winner from the dictionaries (stored in list fs_ranks_list)    
        for l in fs_ranks_list:
            del l[vote_winner]
    return fs_winners[0:req_winners]


def voting_STV(features_ranking, req_winners):
    """
    Single transferable vote method
    https://en.wikipedia.org/wiki/Single_transferable_vote
    Parameters
    ----------
    features_ranking: 2d np array, each row contains 1 sorted vote.
    each row = one voter.
    int req_winners: how many winners should the method return.
    req_winners: integer, how many winners to return.
    """
    fs_ranks_list = []
    for i in features_ranking:
        fs_ranks_list.append({"count": 1, "ballot": i.tolist()})
    return STV(fs_ranks_list, required_winners=req_winners).as_dict()


def voting_plurality(features_ranking, req_winners=5):
    """
    Voting similiar to plurality.
    Parameters
    ----------
    features_ranking
    req_winners
    Returns
    -------
    list of winners according to plurality method.
    """
    fs_winners = []
    temp = np.zeros((len(features_ranking), len(
        features_ranking[0])))  # create 2d array filled with zeros with dimension of features_ranking array
    for i in np.arange(len(features_ranking[0])):
        temp = np.zeros((len(temp), len(temp[0]) - 1))  # create 2d array, with one less col than temp array.
        most_common = get_most_common(features_ranking[:, 0])
        for j in np.arange(len(features_ranking)):
            # remove the most common number in the array from every row
            temp[j] = np.delete(features_ranking[j], get_index(features_ranking[j], most_common))
        features_ranking = np.copy(temp)
        fs_winners.append(most_common)
    return fs_winners[0:req_winners]


def voting_mean(features_ranking, req_winners):
    """
    Mean method
    Parameters
    ----------
    features_ranking: 2d np array, each row contains 1 sorted vote.
    int req_winners: how many winners should the method return.

    Returns
    -------
    winners according to mean method.
    """
    mean_array = np.zeros(len(features_ranking[0]))
    for i in np.arange(len(features_ranking[0])):
        mean_array[i] = features_ranking[:, i].mean()
    fs_winners = mean_array.argsort()
    return fs_winners[0:req_winners]


def voting_min(features_ranking, req_winners):
    """
    Min method
    Parameters
    ----------
    features_ranking: 2d np array, each row contains 1 sorted vote.
    req_winners: how many winners should the method return.

    Returns
    -------
    winners according to min method.
    """
    min_array = np.zeros(len(features_ranking[0]))
    for i in np.arange(len(features_ranking[0])):
        min_array[i] = features_ranking[:, i].min()
    fs_winners = min_array.argsort()
    return fs_winners[0:req_winners]


def voting_max(features_ranking, req_winners):
    """
    max method
    Parameters
    ----------
    features_ranking: 2d np array, each row contains 1 sorted vote.
    req_winners: how many winners should the method return.

    Returns
    -------
    winners according to max method.
    """
    max_array = np.zeros(len(features_ranking[0]))
    for i in np.arange(len(features_ranking[0])):
        max_array[i] = features_ranking[:, i].max()
    fs_winners = max_array.argsort()
    return fs_winners[0:req_winners]


def voting_median(features_ranking, req_winners):
    """
    median method.
    Parameters
    ----------
    features_ranking: 2d np array, each row contains 1 sorted vote.
    req_winners: how many winners should the method return.

    Returns
    -------
    winners according to median method.
    """
    median_array = np.zeros(len(features_ranking[0]))
    for i in np.arange(len(features_ranking[0])):
        median_array[i] = np.median(features_ranking[:, i])
    fs_winners = median_array.argsort()
    return fs_winners[0:req_winners]


def voting_copeland(features_ranking, req_winners):
    """
    Parameters
    ----------
    features_ranking:
    2d np array, each column contains 1 sorted vote.
    Returns
    -------
    Ordered np.array of winners
    """
    fs_winners = []
    return fs_winners[0:req_winners]


def voting_min_max(features_ranking, req_winners):
    """
    Minmax voting method
    Parameters
    ----------
    features_ranking:
    2d np array, each column contains 1 sorted vote.
    req_winners:
    integer: how many winners to return.
    Returns
    -------
    Orderer np.array of length req_winners of winners.
    """
    fs_winners = []
    return fs_winners[0:req_winners]


def voting_kemeny_young(features_ranking, req_winners):
    """
    Kemeny Young optimal rank aggregation.
    copied from http://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html
    Parameters
    ----------
    features_ranking:
    2d np array, each column contains 1 sorted vote.
    Returns
    -------
    Ordered np.array of winners
    """

    n_voters, n_candidates = features_ranking.shape
    # maximize c.T * x
    edge_weights = _build_graph(features_ranking)
    c = -1 * edge_weights.ravel()
    idx = lambda i, j: n_candidates * i + j
    # constraints for every pair
    pairwise_constraints = np.zeros(((n_candidates * (n_candidates - 1)) / 2,
                                     n_candidates ** 2))
    for row, (i, j) in zip(pairwise_constraints,
                           combinations(range(n_candidates), 2)):
        row[[idx(i, j), idx(j, i)]] = 1
    # and for every cycle of length 3
    triangle_constraints = np.zeros(((n_candidates * (n_candidates - 1) *
                                      (n_candidates - 2)),
                                     n_candidates ** 2))
    for row, (i, j, k) in zip(triangle_constraints,
                              permutations(range(n_candidates), 3)):
        row[[idx(i, j), idx(j, k), idx(k, i)]] = 1
    constraints = np.vstack([pairwise_constraints, triangle_constraints])
    constraint_rhs = np.hstack([np.ones(len(pairwise_constraints)),
                                np.ones(len(triangle_constraints))])
    constraint_signs = np.hstack([np.zeros(len(pairwise_constraints)),  # ==
                                  np.ones(len(triangle_constraints))])  # >=

    obj, x, duals = lp_solve(c, constraints, constraint_rhs, constraint_signs,
                             xint=range(1, 1 + n_candidates ** 2))

    x = np.array(x).reshape((n_candidates, n_candidates))
    aggr_rank = x.sum(axis=1)
    return aggr_rank[0:req_winners]


def voting_own_borda(features_ranking, req_winners=5):
    """
    Borda voting method
    Parameters
    ----------
    features_ranking
    2d np array, each column contains 1 sorted vote.
    req_winners
    number of winners, which should borda return
    Returns
    -------
    top winners
    """
    n_of_voters = len(features_ranking) - 1  # number of voters
    features_score = np.zeros(len(features_ranking[0]))  # number in nth index is score for nth feature.
    for voter in features_ranking:
        for score_to_add, feat in enumerate(np.arange(n_of_voters, -1, -1)):  # feat iters from n_of_voters to zero
            features_score[voter[feat]] += score_to_add  # add score to actual score of feat
    return np.argsort(features_score)[::-1][0:req_winners]


def voting_borda_weighted(features_ranking, req_winners=5, N=5, mode='step'):
    """
    Weighted Borda voting method with 
    Parameters
    ----------
    features_ranking
    2d np array, each column contains 1 sorted vote.
    req_winners
    number of winners, which should borda return
    mode: which coefficient calculator should we use
    n: optional parameter for step coefficient calculator
    Returns
    -------
    top winners
    """
    n_of_voters = len(features_ranking) - 1  # number of voters
    features_score = np.zeros(len(features_ranking[0]))  # number in nth index is score for nth feature.
    for voter in features_ranking:
        for score_to_add, feat in enumerate(np.arange(n_of_voters, -1, -1)):  # feat iters from n_of_voters to zero
            features_score[voter[feat]] += score_to_add * get_coeff(score_to_add, mode=mode,
                                                                    N=N)  # add score to actual score of feat
    return np.argsort(features_score)[::-1][0:req_winners]
