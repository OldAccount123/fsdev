import numpy as np
from scipy import stats
from sklearn.feature_selection import f_classif
from minepy import MINE
from skfeature.function.statistical_based import gini_index
from skfeature.function.similarity_based import reliefF, fisher_score, trace_ratio
from skfeature.function.sparse_learning_based import ll_l21, RFS
from skfeature.utility import sparse_learning


def fs_ttest(X_train, y_train):
    """
    t-test feature selection
    
    version
    1. (16.1.2016) starting version ml_featureSelection (v2). Change sorting to sort
    according the p-value (results should be the same as for feature score). Removed 
    unnecessary code. 
    """

    N_feats = X_train.shape[1]
    # featureScore = np.zeros(N_feats)
    pScore = np.zeros(N_feats)

    for ii in np.arange(0, N_feats):
        lll = y_train == y_train[0]  # class labels, true false
        feature_vec = X_train[:, ii]
        feature_class0 = feature_vec[lll]  # take features of first class only
        lll[:] = [not i for i in lll]  # negation
        feature_class1 = feature_vec[lll]  # take features of the second class
        two_sample = stats.ttest_ind(feature_class0, feature_class1)
        # featureScore[ii] = abs(two_sample[0])
        pScore[ii] = two_sample[1]

    sorted_features = np.argsort(pScore)  # [::-1]

    return sorted_features


def fs_bdist(X_train, y_train):
    """
    batachrya distance based feature selection
    
    version
    1. (16.1.2016) starting version ml_featureSelection (v2). Removed 
    unnecessary code. 
    
    only for binary data   # [1] Theodoridis, S. and Koutroumbas, K.  (1999) Pattern Recognition,
    Academic Press, pp. 341-342., pp 152. -matlab biotoolvbox implement
     
    """
    N_feats = X_train.shape[1]
    featureScore = np.zeros(N_feats)

    lll = y_train == y_train[0]  # class labels, true false

    feature_c0 = X_train[lll, :]
    lll[:] = [not i for i in lll]  # negation
    feature_c1 = X_train[lll, :]

    c1_mean = np.mean(feature_c1, axis=0)
    c0_mean = np.mean(feature_c0, axis=0)
    c0_cov = np.var(feature_c0, axis=0)
    c1_cov = np.var(feature_c1, axis=0)

    s1 = np.sqrt(c1_cov)
    s0 = np.sqrt(c0_cov)

    bscore = np.divide(np.power((c1_mean - c0_mean), 2), np.divide((s1 + s0), 4))
    + np.divide(np.log(np.divide((s1 + s0), np.divide(np.multiply(s1, s0), 2))), 2)

    featureScore = abs(bscore)
    sorted_features = np.argsort(featureScore)[::-1]

    return sorted_features


def fs_anova(X_train, y_train):
    """
    ANOVA based feature selection
    
    version
    1. (19.1.2016) starting version ml_featureSelection(v2). Removed 
    unnecessary code.     
         
    """

    F, pval = f_classif(X_train, y_train)
    sorted_features = np.argsort(pval)

    return sorted_features


def fs_pearson(X, y):
    """
    Pearson correlation based feature selection
    
    version
    1. (31.1.2016) starting version         
    """

    N_feats = X.shape[1]
    feature_score = np.zeros(N_feats)
    p_value = np.zeros(N_feats)

    for ii in np.arange(0, N_feats):
        corr_result = stats.pearsonr(X[:, ii], y)
        feature_score[ii] = abs(corr_result[0])
        p_value[ii] = corr_result[1]

    sorted_features = np.argsort(feature_score)[::-1]

    return sorted_features


def fs_mic(X, y):
    """
    Mutual Information Coefficient based feature selection
    
    version
    1. (31.1.2016) starting version 
         
    """
    m = MINE()
    N_feats = X.shape[1]
    feature_score = np.zeros(N_feats)

    for ii in np.arange(0, N_feats):
        m.compute_score(X[:, ii], y)
        feature_score[ii] = m.mic()

    sorted_features = np.argsort(feature_score)[::-1]

    return sorted_features


def fs_relief(X, y):
    """
    relief feature selection
    Robnik-Sikonja, Marko et al. "Theoretical and empirical analysis of relieff and rrelieff." Machine Learning 2003.
    Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.
    http://featureselection.asu.edu/html/skfeature.function.similarity_based.reliefF.html
    
    version
    1. (31.1.2016) starting version         
    """

    score = reliefF.reliefF(X, y)
    sorted_features = np.argsort(score)[::-1]

    return sorted_features


def fs_gini(X, y):
    """
    relief feature selection
    Robnik-Sikonja, Marko et al. "Theoretical and empirical analysis of relieff and rrelieff." Machine Learning 2003.
    Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.
    http://featureselection.asu.edu/html/skfeature.function.similarity_based.reliefF.html
    
    version
    1. (31.1.2016) starting version         
    """

    score = gini_index.gini_index(X, y.astype(int))
    sorted_features = np.argsort(score)

    return sorted_features


def fs_fischer(X, y):
    """
    relief feature selection
    Robnik-Sikonja, Marko et al. "Theoretical and empirical analysis of relieff and rrelieff." Machine Learning 2003.
    Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.
    http://featureselection.asu.edu/html/skfeature.function.similarity_based.reliefF.html
    
    version
    1. (31.1.2016) starting version         
    """

    score = fisher_score.fisher_score(X, y.astype(int))
    sorted_features = np.argsort(score)[::-1]

    return sorted_features


def fs_trace_ration(X, y, n, **kwargs):
    """
    trace ratio criterion for feature selection
    Feiping Nie et al. "Trace Ratio Criterion for Feature Selection." AAAI 2008.

    version
    1. (29.08.2016) starting version
    """

    sorted_features, feature_score, subset_score = trace_ratio.trace_ratio(X, y, n, **kwargs)
    return sorted_features


def fs_ll21(X, y, gamma, **kwargs):
    """
    supervised sparse feature selection via l2,1 norm, i.e.,
    min_{W} sum_{i}log(1+exp(-yi*(W'*x+C))) + z*||W||_{2,1}
    Liu, Jun, et al. "Multi-Task Feature Learning Via Efficient l2,1-Norm Minimization." UAI. 2009.

    version
    1. (30.08.2016) starting version
    """

    Y = sparse_learning.construct_label_matrix_pan(y)
    weight, obj, value_gamma = ll_l21.proximal_gradient_descent(X, Y, gamma)
    sorted_features = sparse_learning.feature_ranking(weight)
    return sorted_features


def fs_RFS(X, y, **kwargs):
    """
    efficient and robust feature selection via joint l21-norms minimization
    min_W||X^T W - Y||_2,1 + gamma||W||_2,1
    Nie, Feiping et al. "Efficient and Robust Feature Selection via Joint l2,1-Norms Minimization" NIPS 2010.
    
    1. (31.08.2016) starting version 
    """
    Y = sparse_learning.construct_label_matrix(y)
    weight = RFS.rfs(X, Y, **kwargs)
    sorted_features = sparse_learning.feature_ranking(weight)
    return sorted_features
