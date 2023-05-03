import numpy as np

from ccgaussian.test.stats import cluster_acc


def bootstrap_metric(y_pred, y_true, metric_func,
                     n_bootstraps=200, rng_seed=123):
    """Compute test set boostrapping of a metric

    Args:
        y_pred (np.array): Model predictions for some output y
        y_true (np.array): True value of output y
        metric_func (function): function with parameters (y_pred, y_true)
                                returning a float metric
        n_bootstraps (int, optional): Number of bootstrap samples to take.
                                      Defaults to 200.
        rng_seed (int, optional): Random seed for reproducibility.
                                  Defaults to 123.

    Returns:
        tuple: metric_mean: float with bootstrapped mean of metric
               ci_low: Low value from 95% confidence interval
               ci_high: High value from 95% confidence interval
    """
    # set bootstrap sample size and random seed
    n_bootstraps = n_bootstraps
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    idx = np.arange(y_pred.shape[0])
    # bootstrap
    for _ in range(n_bootstraps):
        sample_idx = rng.choice(idx, size=y_pred.shape[0], replace=True)
        score = metric_func(y_pred[sample_idx], y_true[sample_idx])
        bootstrapped_scores.append(score)
    # compute mean and confidence interval
    metric_mean = np.mean(bootstrapped_scores)
    ci_low = np.percentile(bootstrapped_scores, 2.5)
    ci_high = np.percentile(bootstrapped_scores, 97.5)
    return (metric_mean, ci_low, ci_high)


def bootstrap_gcd_acc(y_pred, y_true, norm_mask,
                      n_bootstraps=200, rng_seed=123):
    """Compute test set boostrapping of a metric

    Args:
        y_pred (np.array): Model predictions for some output y
        y_true (np.array): True value of output y
        norm_mask   (np.array): Mask for which y_true are normal
        n_bootstraps (int, optional): Number of bootstrap samples to take.
                                      Defaults to 200.
        rng_seed (int, optional): Random seed for reproducibility.
                                  Defaults to 123.

    Returns:
        tuple: metric_mean: float with bootstrapped mean of metric
               ci_low: Low value from 95% confidence interval
               ci_high: High value from 95% confidence interval
    """
    # set bootstrap sample size and random seed
    n_bootstraps = n_bootstraps
    bootstrapped_scores = np.empty((0, 3))
    rng = np.random.RandomState(rng_seed)
    idx = np.arange(y_pred.shape[0])
    # determine normal classes
    norm_classes = np.unique(y_true[norm_mask])
    # bootstrap
    for _ in range(n_bootstraps):
        sample_idx = rng.choice(idx, size=y_pred.shape[0], replace=True)
        score = cluster_acc(y_pred[sample_idx], y_true[sample_idx], norm_classes)
        bootstrapped_scores = np.vstack((bootstrapped_scores, score))
    # compute mean and confidence interval
    mean = np.mean(bootstrapped_scores, axis=0)
    ci_low = np.percentile(bootstrapped_scores, 2.5, axis=0)
    ci_high = np.percentile(bootstrapped_scores, 97.5, axis=0)
    return (mean, ci_low, ci_high)
