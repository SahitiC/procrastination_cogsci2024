import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3
mpl.rc('text', usetex=False)

# %%
# legends


def legends():

    custom_lines = [Line2D([0], [0], color='hotpink', lw=2),
                    Line2D([0], [0], color='mediumturquoise', lw=2),
                    Line2D([0], [0], color='goldenrod', lw=2)]

    fig, ax = plt.subplots(figsize=(1, 1), dpi=300, bbox_inches='tight')
    ax.axis('off')
    ax.legend(custom_lines, ['0.6',
                             '0.3',
                             '0.2'], title=r'$\eta_{assumed}$',
              frameon=False)

# %%
# plot trajectories


def sausage_plots(trajectories, color, horizon):
    mean = np.mean(trajectories, axis=0)
    error = np.std(trajectories, axis=0)
    plt.plot(mean, color=color)
    plt.fill_between(np.arange(horizon+1),
                     mean-error, mean+error,
                     alpha=0.1, color=color)


def example_trajectories(trajectories, color, lwidth, number):

    for i in range(number):
        plt.plot(trajectories[i], color=color,
                 linewidth=lwidth, linestyle='dashed')
