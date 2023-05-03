import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_con_matrix(con_matrix):
    disp = metrics.ConfusionMatrixDisplay(con_matrix)
    disp = disp.plot(cmap="Blues", values_format=".0f")
    return disp.figure_


def plot_gcd_ci(means, ci_low, ci_high):
    """Plot confidence interval for GCD results
    Returns:
        plt.Figure: Figure with confidence interval plot
    """
    fig, ax = plt.subplots()
    x = np.arange(3)
    ax.bar(x, means, yerr=[means - ci_low, ci_high - means], capsize=5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["All", "Normal", "Novel"])
    ax.set_ylabel("Accuracy")
    return fig


def plot_md_hist(embeds, targets, means, sigma2s):
    fig, ax = plt.subplots()
    num_embeds = embeds.shape[0]
    md = ((embeds - means[targets])**2 / sigma2s).sum(axis=1)**(.5)
    ax.hist(md, weights=np.array([1 / num_embeds] * num_embeds))
    ax.set_xlabel("Mahalanobis Distance")
    ax.set_ylabel("Distance Frequency")
    return fig


def plot_md_scatter(embeds, targets, means, sigma2s):
    fig, ax = plt.subplots()
    num_embeds = embeds.shape[0]
    # get Mahalanobis distance and cosine similarity for points
    md = ((embeds - means[targets])**2 / sigma2s).sum(axis=1)**(.5)
    cos_sim = np.empty((num_embeds,))
    for target in np.unique(targets):
        tar_mask = targets == target
        cos_sim[tar_mask] = embeds[tar_mask] @ means[target] / \
            (np.linalg.norm(embeds[tar_mask], axis=1) * np.linalg.norm(means[target]))
    ax.scatter(md, cos_sim, alpha=.1)
    ax.set_xlabel("Mahalanobis Distance")
    ax.set_ylabel("Cosine Similarity")
    return fig
