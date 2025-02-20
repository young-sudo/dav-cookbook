#!/usr/bin/env python3

import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

path = os.getcwd()
parent = os.path.dirname(path) # extract parent dir in cross platform way

path_to_plots = os.path.join(parent, "plots")
path_to_data = os.path.join(parent, "data")


if __name__ == "__main__":

    train_clean = pd.read_csv(os.path.join(path_to_data, "train_cleaned.csv"))

    # Correlation with label (Survived)
    print(train_clean.corr(method='pearson', numeric_only=True)["Survived"])

    # Correlation table
    # corr = train_clean.corr(numeric_only=True)
    # print(corr)
    # corr.style.background_gradient(cmap='coolwarm')

    # Pairplot
    # Histograms on diagonals and scatterplots for features
    def corrfunc(x, y, ax=None, **kws):
        """Plot the correlation coefficient in the corner of a plot."""
        # https://stackoverflow.com/questions/50832204/show-correlation-values-in-pairplot
        r, _ = pearsonr(x, y)
        ax = ax or plt.gca()
        ax.annotate(f'ρ = {r:.2f}', xy=(.7, .8), xycoords=ax.transAxes)

    g = sns.pairplot(train_clean)
    g.map_lower(corrfunc)
    plt.suptitle("Pairplot for features with correlation values and histograms for each feature", x=0.5, y=1)
    # plt.show()

    plot_path = os.path.join(path_to_plots, "corr_plot.png")

    print()
    plt.savefig(plot_path)
    print(f"Pairplot saved to: {plot_path}")

