import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def binned_scatter_plot_with_error(x, y, bins=20, scatter_alpha=0.2,
                                    show_original=True, xlabel='x', ylabel='y', title=None):
    """
    Create a binned scatter plot with mean and standard deviation error bars.
    NaN values are ignored.

    Parameters:
        x (array-like): The x-values of the data points.
        y (array-like): The y-values of the data points.
        bins (int or sequence): Number of bins or bin edges.
        scatter_alpha (float): Opacity of the raw scatter points.
        show_original (bool): Whether to show the original scatter points.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Compute mean and std per bin
    means, bin_edges, binnumber = binned_statistic(x_clean, y_clean, statistic='mean', bins=bins)
    stds, _, _ = binned_statistic(x_clean, y_clean, statistic='std', bins=bin_edges)

    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8, 5))

    if show_original:
        plt.scatter(x_clean, y_clean, alpha=scatter_alpha, s=10, label='Raw data')

    # Plot binned means with error bars (std)
    plt.errorbar(bin_centers, means, yerr=stds, fmt='o-', color='red', ecolor='gray',
                 capsize=4, label='Binned mean ± std')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
