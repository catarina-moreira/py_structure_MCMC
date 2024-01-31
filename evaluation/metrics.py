
import numpy as np
from scipy.stats import entropy


##
## kl_divergence
###########################################################################################
def kl_divergence(P : dict, Q : dict):
    """computes the KL divergence between two distributions. Requires that the two distributions have the same length

    Args:
        P (dict): distribution 1
        Q (dict): distribution 2

    Returns:
        float: KL divergence
    """
    # Ensure that the distributions have the same length
    # if the dist1 and dist2 do not have the same length, return an error and exit
    len_diff = np.abs(len(P) - len(Q))
    if len_diff > 0:
        print(f"Error: dist1 and dist2 have different lengths: {len(P)} and {len(Q)}")
        return -1
    
    return entropy(P, Q)


##
## jensen_shannon_divergence
###########################################################################################
def jensen_shannon_divergence(P: dict, Q: dict, tol: float = 1e-15):
    """_summary_

    Args:
        dist1 (dict): _description_
        dist2 (dict): _description_

    Returns:
        _type_: _description_
    """
    # Ensure the distributions have the same length
    len_diff = np.abs(len(P) - len(Q))
    if len_diff > 0:
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return -1
    
    # Convert the distributions to lists (ensuring consistent order)
    p = np.array(list(P.values())) + tol
    q = np.array(list(Q.values())) + tol

    # Normalize the distributions to ensure they are proper probability distributions
    p /= p.sum()
    q /= q.sum()

    # Compute M
    m = 0.5 * (p + q)
    
    # Compute the Jensen-Shannon divergence
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    
    return jsd


def mean_squared_error(res, true_res):
    mse = 0
    for g in res.keys():
        mse += (res[g] - true_res[g])**2
    return mse


def mean_absolute_error(res, true_res):
    mae = 0
    for g in res.keys():
        mae += np.abs(res[g] - true_res[g])
    return mae


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.
    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition
    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG
    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """

    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'tp': len(true_pos), 'fp': float(len(reverse) + len(false_pos)), 'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd}