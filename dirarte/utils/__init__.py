import numpy as np
import numba
from scipy.stats import median_abs_deviation
from scipy.stats import gaussian_kde as kde
from scipy.interpolate import interp1d
from sklearn.metrics import adjusted_rand_score
from joblib import Parallel, delayed
from pyscipopt import quicksum

from .tree import (
    HAS_LGB,
    HAS_XGB,
    parse_trees,
    is_boosted_trees,
    get_intercept,
)

MAX_VAL, MIN_VAL = 1e+8, -1e+8
Y_TARGET = 1


"""
Utility functions for cost functions
"""

def compute_percentiles(X, is_immutable, l_buff=1e-6, r_buff=1e-6, l_quantile=0.001, r_quantile=0.999, grid_size=100):
    """ Compute percentile functions for each feature using KDE. """
    
    percentile = []
    for d in range(X.shape[1]):
        if is_immutable[d]:
            percentile.append(None)
            continue
        kde_estimator = kde(X[:, d])
        grid = np.linspace(np.quantile(X[:, d], l_quantile), np.quantile(X[:, d], r_quantile), grid_size)
        pdf = kde_estimator(grid)
        cdf_raw = np.cumsum(pdf)
        total = cdf_raw[-1] + l_buff + r_buff
        cdf = (l_buff + cdf_raw) / total
        p_d = interp1d(x=grid, y=cdf, copy=False, fill_value=(l_buff, 1.0 - r_buff), bounds_error=False, assume_sorted=False)
        percentile.append(p_d)
    return percentile


def compute_weights(X, cost_type, is_binary):
    """ Compute feature weights based on the specified cost type. """
    
    weight = np.ones(X.shape[1])
    if cost_type == 'mad':
        weight = (median_abs_deviation(X) + 1e-4) ** -1
        weight[is_binary] = (X[:, is_binary] * 1.4826).std(axis=0)
    elif cost_type == 'std':
        weight = np.std(X, axis=0) ** -1
        weight[is_binary] = (X[:, is_binary] * 1.4826).std(axis=0)
    elif cost_type == 'normalized':
        weight = (X.max(axis=0) - X.min(axis=0)) ** -1
    return weight



"""
Utility functions for search-based methods
"""

def prune_trees(estimator, X, n_select, n_jobs=-1):
    """ Prune trees in the ensemble by selecting a diverse subset based on the adjusted Rand index of their leaf assignments. """
    
    Z = estimator.apply(X)
    n_trees = Z.shape[1]
    triu_indices = np.triu_indices(n_trees, k=1)
    
    def compute_ari(i, j):
        return adjusted_rand_score(Z[:, i], Z[:, j])

    ari_values = Parallel(n_jobs=n_jobs)(
        delayed(compute_ari)(i, j) for i, j in zip(*triu_indices)
    )
    S = np.ones((n_trees, n_trees))
    S[triu_indices] = ari_values
    S.T[triu_indices] = ari_values
    
    selected = []
    first_pair = np.unravel_index(np.argmin(S), S.shape)
    selected.extend(first_pair)    

    max_sims = np.maximum(S[:, selected[0]], S[:, selected[1]])    
    mask = np.ones(n_trees, dtype=bool)
    mask[selected] = False

    while len(selected) < n_select:
        remaining_indices = np.where(mask)[0]
        best_tree = remaining_indices[np.argmin(max_sims[remaining_indices])]        
        selected.append(best_tree)
        mask[best_tree] = False
        max_sims = np.maximum(max_sims, S[:, best_tree])
        
    return np.array(selected)   


@numba.njit("float64[:, :, :](int64[:], float64[:], int64[:], int64[:], int64, int64)", cache=True)
def compute_regions(feature, threshold, children_left, children_right, n_features_in, node_count):
    """ Compute the region corresponding to each node in the tree. """

    R = np.zeros((node_count, n_features_in, 2), dtype=np.float64)
    R[:, :, 0] = MIN_VAL
    R[:, :, 1] = MAX_VAL
    for j in range(node_count):
        if feature[j] >= 0:
            R[children_left[j]] = R[j]
            R[children_right[j]] = R[j]
            R[children_left[j], feature[j], 1] = threshold[j]
            R[children_right[j], feature[j], 0] = threshold[j]
    return R


@numba.njit("float64[:, :, :](float64[:, :, :], float64[:, :, :], float64[:], float64[:])", parallel=True, cache=True)
def merge_regions(regions1, regions2, values1, values2):
    """ Merge two sets of regions by computing their intersections. """

    n_features = regions1.shape[1]
    n_regions1, n_regions2 = regions1.shape[0], regions2.shape[0]
    R = np.zeros((n_regions1 * n_regions2, n_features + 1, 2), dtype=np.float64)
    is_nonempty = np.zeros(n_regions1 * n_regions2, dtype=np.bool_)
    for i in numba.prange(n_regions1):    
        for j in numba.prange(n_regions2):
            k = i * n_regions2 + j
            R[k, :-1, 0] = np.maximum(regions1[i, :, 0], regions2[j, :, 0])
            R[k, :-1, 1] = np.minimum(regions1[i, :, 1], regions2[j, :, 1])
            R[k, -1, 0] = values1[i]
            R[k, -1, 1] = values2[j]
            is_nonempty[k] = np.all(R[k, :-1, 0] < R[k, :-1, 1])
    R = R[is_nonempty]
    return R


@numba.njit("float64[:, :, :](float64[:, :], float64[:, :, :], float64[:], boolean[:], boolean[:])", parallel=True, cache=True)
def compute_all_actions(X, regions, feature_steps, is_binary, is_integer):
    """ Compute all possible actions for recourse. """
    
    A = np.zeros((X.shape[0], regions.shape[0], X.shape[1]), dtype=np.float64)
    for i in numba.prange(X.shape[0]):
        for l in numba.prange(regions.shape[0]):
            for d in numba.prange(X.shape[1]):
                if X[i, d] <= regions[l][d][0]:
                    if is_binary[d]:
                        A[i, l, d] = 1.0
                    elif is_integer[d]:
                        A[i, l, d] = np.ceil(regions[l][d][0]) - X[i, d] + feature_steps[d]
                    else:
                        A[i, l, d] = regions[l][d][0] - X[i, d] + 1e-8 + feature_steps[d]
                elif X[i, d] <= regions[l][d][1]:
                    A[i, l, d] = 0.0
                else:
                    if is_binary[d]:
                        A[i, l, d] = - 1.0
                    elif is_integer[d]:
                        A[i, l, d] = np.floor(regions[l][d][1]) - X[i, d] - feature_steps[d]
                    else:
                        A[i, l, d] = regions[l][d][1] - X[i, d] - feature_steps[d]
    return A


@numba.njit("boolean[:, :](float64[:, :, :], boolean[:], boolean[:], boolean[:], int64)", parallel=True, cache=True)
def is_feasible(A, is_immutable, is_unincreasable, is_irreducible, max_features):
    """ Check feasibility of actions based on constraints. """
    
    F = np.ones((A.shape[0], A.shape[1]), dtype=np.bool_)
    is_fix = (is_immutable.sum() > 0)
    is_inc = (is_unincreasable.sum() > 0)
    is_red = (is_irreducible.sum() > 0)
    for i in numba.prange(A.shape[0]):
        if is_fix:
            F[i] = F[i] * (np.count_nonzero(A[i][:, is_immutable], axis=1) == 0)
        if is_inc:
            F[i] = F[i] * (np.count_nonzero(np.clip(A[i][:, is_unincreasable], 0.0, None), axis=1) == 0)
        if is_red:
            F[i] = F[i] * (np.count_nonzero(np.clip(A[i][:, is_irreducible], None, 0.0), axis=1) == 0)
        if max_features > 0: 
            F[i] = F[i] * (np.count_nonzero(A[i], axis=1) <= max_features)
    return F


@numba.njit("float64[:, :, :](float64[:, :], float64[:, :, :], boolean[:, :, :], float64[:, :])", parallel=True, cache=True)
def find_best_actions(X, A, F, O):
    """ Find the best action for each threshold and instance based on feasibility and objective value. """
    
    OA = np.zeros((F.shape[0], X.shape[0], 1 + X.shape[1]), dtype=np.float64)
    O_opt = np.ones((F.shape[0], X.shape[0]), dtype=np.float64) * MAX_VAL
    for t in numba.prange(F.shape[0]):
        for i in numba.prange(X.shape[0]):
            if F[t, i].sum() == 0: continue
            for l in range(A.shape[1]):
                if F[t, i, l] and (O[i, l] < O_opt[t, i]):
                    OA[t, i, 0] = O[i, l]
                    OA[t, i, 1:] = A[i, l]
                    O_opt[t, i] = O[i, l]
    return OA



"""
Utility functions for MILO-based methods
"""

def flatten(x): 
    """ Flatten a list of lists into a single list. """
    
    return sum(x, [])
        
        
def LinSum(Vars): 
    """ Compute the linear sum of a list of variables. """
    
    return quicksum(Vars)


def LinExpr(Coeffs, Vars): 
    """ Compute the linear expression given coefficients and variables. """

    return quicksum(Coeffs[i] * Vars[i] for i in range(len(Coeffs)))


@numba.njit("boolean[:, :](float64[:, :], float64[:, :, :], boolean)", parallel=True, cache=True)
def compute_leaf_indicators(cf_values, regions, is_xgboost):
    """ Compute indicators for whether counterfactual values fall within the corresponding regions. """
    
    I = np.zeros((regions.shape[0], cf_values.shape[0]), dtype=np.bool_)
    for l in numba.prange(regions.shape[0]):
        for i in numba.prange(cf_values.shape[0]):
            if is_xgboost:
                if (regions[l, int(cf_values[i, 0]), 0] <= cf_values[i, 1]) and (cf_values[i, 1] < regions[l, int(cf_values[i, 0]), 1]):
                    I[l, i] = True
            else:
                if (regions[l, int(cf_values[i, 0]), 0] < cf_values[i, 1]) and (cf_values[i, 1] <= regions[l, int(cf_values[i, 0]), 1]):
                    I[l, i] = True
    return I