import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import ast
from tslearn.clustering import TimeSeriesKMeans
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 3


# %%
# pre-processing and clustering
# already clustered data is saved as csv, no need to run this agian

data = pd.read_csv('FollowUpStudymatrixDf_finalpaper.csv')

data_relevant = data.dropna(subset=['delta progress'])
data_relevant = data_relevant.reset_index(drop=True)

# drop the ones that discontinued (subj. 1, 95, 111)
# sbj 24, 55, 126 also dont finish 7 hours but not because they drop out
data_relevant = data_relevant.drop([1, 90, 104])
data_relevant = data_relevant.reset_index(drop=True)

# normalised cumulative sequence
cumulative_normalised = []
for i in range(len(data_relevant)):
    temp = np.array(
        ast.literal_eval(data_relevant['cumulative progress'][i]))
    cumulative_normalised.append(temp/data_relevant['Total credits'][i])
data_relevant['cumulative progress normalised'] = cumulative_normalised

# represent delta progress in weeks
semester_length = len(ast.literal_eval(data_relevant['delta progress'][0]))
semester_length_weeks = round(semester_length/7)
delta_progress_weeks = []
plt.figure()

# delta progress week wise
for i in range(len(data_relevant)):

    temp = ast.literal_eval(data_relevant['delta progress'][i])
    temp_week = []
    for i_week in range(semester_length_weeks):

        temp_week.append(
            sum(temp[i_week*7: (i_week+1)*7]) * 1.0)

    assert sum(temp_week) == data_relevant['Total credits'][i]
    delta_progress_weeks.append(temp_week)
    plt.plot(temp_week)
data_relevant['delta_progress_weeks'] = delta_progress_weeks

# get cumulative progress from delta progress
cumulative_progress_weeks = []
for i in range(len(data_relevant)):

    cumulative_progress_weeks.append(
        list(np.cumsum(
            ast.literal_eval(data_relevant['delta_progress_weeks'][i])))
    )
data_relevant['cumulative_progress_weeks'] = cumulative_progress_weeks

# inertia vs cluster number
inertia = []
for cluster_size in range(15):
    km = TimeSeriesKMeans(n_clusters=cluster_size+1, n_init=5,
                          metric="euclidean", verbose=True)
    timseries_to_cluster = np.vstack(
        data_relevant['cumulative progress normalised'])
    labels = km.fit_predict(timseries_to_cluster)
    inertia.append(km.inertia_)

plt.plot(inertia)
plt.xticks(np.arange(15), labels=np.arange(1, 16))
plt.xlabel('cluster number')
plt.ylabel('k-means sum of squares')

# pick best cluster number (elbow method) and cluster again
km = TimeSeriesKMeans(n_clusters=8, n_init=5, metric="euclidean", verbose=True)
timseries_to_cluster = np.vstack(
    data_relevant['cumulative progress normalised'])
labels = km.fit_predict(timseries_to_cluster)
data_relevant['labels'] = labels

# %%
# load clustered data

data_relevant = pd.read_csv('data_relevant_clustered.csv')

# plot clustered data

for label in set(data_relevant['labels']):
    plt.figure(figsize=(4, 4), dpi=300)

    for i in range(len(data_relevant)):

        if data_relevant['labels'][i] == label:
            # ast.literal_eval(data_relevant['delta progress'][i])
            # data_relevant['cumulative progress normalised'][i]
            plt.plot(
                np.array(ast.literal_eval(
                    data_relevant['cumulative_progress_weeks'][i])) * 2,
                alpha=0.5)
    sns.despine()
    plt.xlabel('time (weeks)')
    plt.xticks([0, 7, 15])
    plt.yticks(list(plt.yticks()[0][1:-1]) + [14])  # add tick at threshold=14
    plt.ylabel('research units \n completed')
    plt.savefig(
        f'plots/vectors/cluster_{label}.svg',
        format='svg', dpi=300)
