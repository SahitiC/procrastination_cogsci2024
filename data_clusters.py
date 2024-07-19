"""
code for reproducing Figure 1;
what it does: pre-processs data, cluster data using k-means (and select k),
plot trajectories cluster-wise
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import ast
from sklearn.cluster import KMeans
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 3

SAVE_ONLY = True

# %%


def normalize_cumulative_progress(row):
    """
    normalise cumulative progress by total credits
    """
    temp = np.array(ast.literal_eval(row['cumulative progress']))
    return temp / row['Total credits']


def process_delta_progress(row, semester_length_weeks):
    """
    aggregate delta progress over days to weeks
    """
    temp = ast.literal_eval(row['delta progress'])
    temp_week = [sum(temp[i_week*7: (i_week+1)*7])
                 for i_week in range(semester_length_weeks)]

    assert sum(temp_week) == row['Total credits']
    return temp_week


def cumulative_progress_weeks(row):
    """
    get cumulative progress in weeks from delta progress in weeks
    """
    return list(np.cumsum(row['delta_progress_weeks']))


# %%
# pre-processing and clustering

data = pd.read_csv('FollowUpStudymatrixDf_finalpaper.csv')

# drop the ones that discontinued (subj. 1, 95, 111)
# sbj 24, 55, 126 also dont finish 7 hours but not because they drop out
data_relevant = data.drop([1, 95, 111])
data_relevant = data_relevant.reset_index(drop=True)

# drop nan entries
data_relevant = data_relevant.dropna(subset=['delta progress'])
data_relevant = data_relevant.reset_index(drop=True)

# how many dropped in total
print(f'Total subjects dropped = {len(data)-len(data_relevant)}')

# get normalised cumulative progress
data_relevant['cumulative progress normalised'] = data_relevant.apply(
    normalize_cumulative_progress, axis=1)

# transform delta progress to weeks from days
semester_length = len(ast.literal_eval(data_relevant['delta progress'][0]))
semester_length_weeks = round(semester_length/7)
data_relevant['delta_progress_weeks'] = data_relevant.apply(
    lambda row: process_delta_progress(row, semester_length_weeks), axis=1)

# get cumulative progress from delta progress
data_relevant["cumulative_progress_weeks"] = data_relevant.apply(
    cumulative_progress_weeks, axis=1)

timeseries_to_cluster = np.vstack(
    data_relevant['cumulative progress normalised'])

# %%
# find and plot inertia vs cluster number
inertia = []
for cluster_size in range(1, 15):

    km = KMeans(n_clusters=cluster_size+1, n_init=3,
                random_state=0)
    labels = km.fit_predict(timeseries_to_cluster)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(inertia)
plt.xticks(np.arange(14), labels=np.arange(1, 15))
plt.xlabel('cluster number')
plt.ylabel('k-means \n sum of squares')

# pick best cluster number (elbow method) and cluster again
km = KMeans(n_clusters=8, verbose=False, random_state=1, n_init=5)
labels = km.fit_predict(timeseries_to_cluster)
data_relevant['labels'] = labels

# save clustered data
data_relevant.to_csv('data_relevant_clustered.csv', index=False)

# %%
# plot clustered data

grouped = data_relevant.groupby('labels')

for label, group in grouped:
    plt.figure(figsize=(4, 4), dpi=300)
    for index, row in group.iterrows():
        cumulative_progress = np.array(
            row["cumulative_progress_weeks"]) * 2
        plt.plot(cumulative_progress, alpha=0.5)
    plt.title(f'Label: {label}')
    sns.despine()
    plt.xlabel('time (weeks)')
    plt.xticks([0, 7, 15])
    plt.yticks(list(plt.yticks()[0][1:-1]) + [14])  # add tick at threshold=14
    plt.ylabel('research units \n completed')
    plt.savefig(
        f'plots/vectors/cluster_{label}.svg',
        format='svg', dpi=300)

if not SAVE_ONLY:
    plt.show()
