import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import ast
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3


# %%

data_relevant = pd.read_csv('data_relevant_clustered.csv')

cumulative_progress_weeks = []
for i in range(len(data_relevant)):

    cumulative_progress_weeks.append(
        np.cumsum(
            ast.literal_eval(data_relevant['delta_progress_weeks'][i])))

data_relevant['cumulative_progress_weeks'] = cumulative_progress_weeks

# %%
for label in set(data_relevant['labels']):
    plt.figure(figsize=(4, 4), dpi=300)

    for i in range(len(data_relevant)):

        if data_relevant['labels'][i] == label:
            # ast.literal_eval(data_relevant['delta progress'][i])
            # data_relevant['cumulative progress normalised'][i]
            plt.plot(data_relevant['cumulative_progress_weeks'][i],
                     alpha=0.5)
    sns.despine()
    plt.xlabel('time (weeks)')
    plt.ylabel('research hours completed')
    plt.savefig(
        f'plots/vectors/cluster_{label}.svg',
        format='svg', dpi=300
    )
