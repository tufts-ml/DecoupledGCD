import numpy as np
import scipy.optimize as optimize


def cluster_acc(y_pred, y_true, norm_classes):
    row_ind, col_ind, weight = _assign_clusters(y_pred, y_true)
    cluster_norm_mask = np.isin(row_ind, norm_classes)
    return np.array([
        _cluster_acc(row_ind, col_ind, weight),
        _cluster_acc(row_ind[cluster_norm_mask], col_ind[cluster_norm_mask], weight),
        _cluster_acc(row_ind[~cluster_norm_mask], col_ind[~cluster_norm_mask], weight)])


def cluster_confusion(y_pred, y_true):
    return _cluster_confusion(*_assign_clusters(y_pred, y_true))


# cluster functions originally based on:
# https://github.com/k-han/AutoNovel/blob/5eda7e45898cf3fbcde4c34b9c14c743082abd94/utils/util.py#L19\
# but updated to reflect newer methodology:
# https://github.com/sgvaze/generalized-category-discovery/blob/831a645c3d09a68ec4633a45741025765bacf7e0/project_utils/cluster_and_log_utils.py#L29


def _assign_clusters(y_pred, y_true):
    """Calculate cluster assignments

    Args:
        y_true (np.array): true labels (N,)
        y_pred (np.array): predicted labels (N,)

    Returns:
        tuple: np.array of sorted row indices (true) and one of corresponding column indices (pred)
               giving the optimal assignment.
               np.array of weight matrix used for assignment.
    """
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    # add 1 because of zero indexing
    d = max(y_pred.max(), y_true.max()) + 1
    # compute weight matrix with row index as prediction and col index as true
    weight = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        weight[y_true[i], y_pred[i]] += 1
    # compute assignments
    row_ind, col_ind = optimize.linear_sum_assignment(weight, maximize=True)
    return row_ind, col_ind, weight


def _cluster_acc(row_ind, col_ind, weight):
    """Compute clustering accuracy

    Args:
        row_ind (np.array): True class index
        col_ind (np.array): Predicted class index
        weight (np.array): Weight matrix used for assigning clusters

    Returns:
        float: accuracy, in [0,1]
    """
    return float(weight[row_ind, col_ind].sum()) / weight[:, col_ind].sum()


def _cluster_confusion(row_ind, col_ind, weight):
    """Reorder weight matrix to get clustering confusion matrix

    Args:
        row_ind (np.array): True class index
        col_ind (np.array): Predicted class index
        weight (np.array): Weight matrix used for assigning clusters

    Returns:
        np.array: clustering confusion matrix, with true labels as rows
    """
    # add 1 because of zero indexing
    d = max(col_ind) + 1
    # reorder weights according to cluster assignments
    con_matrix = np.zeros((d, d))
    for i, j in zip(row_ind, col_ind):
        con_matrix[:, i] = weight[:, j]
    return con_matrix
