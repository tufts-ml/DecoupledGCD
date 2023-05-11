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


def plot_gcd_confusion(cluster_confusion, y_true, norm_mask):
    """Plot reduced confusion matrix for GCD

    Args:
        cluster_confusion (np.array): Confusion matrix with true labels as rows.
        y_true (np.array): True value of output y
        norm_mask (np.array): Mask for which y_true are normal

    Returns:
        plt.Figure: Figure with reduced confusion matrix for GCD
    """
    # determine normal classes, then clean up variables
    norm_classes = np.unique(y_true[norm_mask])
    del norm_mask, y_true
    # construct normal mask for confusion matrix rows
    norm_mask_row = np.isin(np.arange(cluster_confusion.shape[0]), norm_classes)
    # reduce confusion matrix
    gcd_cm = np.zeros((2, 3))
    for i, cm_row in enumerate(cluster_confusion):
        # first row of gcd_cm for normal, second for novel
        gcd_cm_row = 0 if np.isin(i, norm_classes) else 1
        # correct entries on diagonals, then zero to make later calculations easier
        gcd_cm[gcd_cm_row, 0] += cm_row[i]
        cm_row[i] = 0
        # incorrect entries within normal set
        gcd_cm[gcd_cm_row, 1 + gcd_cm_row] += np.sum(cm_row[norm_mask_row])
        # incorrect entries within novel set
        gcd_cm[gcd_cm_row, 2 - gcd_cm_row] += np.sum(cm_row[~norm_mask_row])
    # plot array
    x_ticks = ["Correct", "Intra-Set", "Inter-Set"]
    y_ticks = ["Normal", "Novel"]
    return plot_array(gcd_cm, x_ticks, y_ticks)


def plot_array(array, x_ticks, y_ticks):
    """Plot array with ticks and value text

    Args:
        array (np.array): (N, M) array of values to plot
        x_ticks (list): (M,) ticks for columns
        y_ticks (list): (N,) ticks for rows

    Returns:
        plt.Figure: Figure with array plot
    """
    # plot array as image
    fig, ax = plt.subplots()
    image = ax.imshow(array, cmap="Blues")
    # set up variables for text coloring
    cmap_min, cmap_max = image.cmap(0.0), image.cmap(1.0)  # need floats for proper scaling
    color_thresh = np.mean([array.min(), array.max()])
    # plot text for each array value
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            text_color = cmap_max if array[i, j] < color_thresh else cmap_min
            ax.text(j, i, f"{array[i, j]:0.0f}", color=text_color, ha="center", va="center")
    # plot tick labels at tick positions
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(range(len(y_ticks)))
    ax.set_yticklabels(y_ticks)
    return fig
