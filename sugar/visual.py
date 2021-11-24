"""
Visualization Tools that can be reproducted.

Author: Wang Rui
Date: 2021.03.09
"""
from itertools import product

import matplotlib.pyplot as plt
import numpy as np


def corrplot(cm=True, x_labels=None, y_labels=None, cmap=plt.cm.YlGnBu_r,
             xticks_rotation='horizontal', values_format=None,
             ax=None, fig=None, colorbar=False, fontsize=8):
    """
    Plot correlation between two series of variables.
        Parameters
        ----------
            cm : numpy.ndarray
                Correlation matrix.
            x_labels : list
                Labels of dim 1.
            y_labels : list
                Labels of dim 2.
            cmap : str or matplotlib Colormap, default='viridis'
                Colormap recognized by matplotlib.
            xticks_rotation : {'vertical', 'horizontal'} or float, \
                Rotation of xtick labels. Default='horizontal'
            values_format : str, default=None
                Format specification for values in confusion matrix. If `None`,
                the format specification is 'd' or '.2g' whichever is shorter.
            ax : matplotlib axes, default=None
                Axes object to plot on. If `None`, a new figure and axes is
                created.
            colorbar : bool, default=True
                Whether or not to add a colorbar to the plot.

        Returns
        -------
            fig
            ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure
    
    n_varsy = cm.shape[0]
    n_varsx = cm.shape[1]
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text = np.empty_like(cm, dtype=object)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_varsy), range(n_varsx)):
        color = cmap_max if cm[i, j] < thresh else cmap_min

        if values_format is None:
            text_cm = format(cm[i, j], '.2g')
            if cm.dtype.kind != 'f':
                text_d = format(cm[i, j], 'd')
                if len(text_d) < len(text_cm):
                    text_cm = text_d
        else:
            text_cm = format(cm[i, j], values_format)

        text[i, j] = ax.text(
            j, i, text_cm, fontsize=fontsize,
            ha="center", va="center",
            color=color)

    x_labels = np.arange(n_varsx) if x_labels is None else x_labels
    y_labels = np.arange(n_varsy) if y_labels is None else y_labels
    if colorbar:
        fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_varsx),
           yticks=np.arange(n_varsy),
           xticklabels=x_labels,
           yticklabels=y_labels,
           xlim=(-0.5, n_varsx - 0.5),
           ylim=(-0.5, n_varsy - 0.5))

    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    return fig, ax
