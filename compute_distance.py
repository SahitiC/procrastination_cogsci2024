"""
module to find average distance between two sets of trajectories;
used to find the average distance between simulated trajectories from a model
and each of the clusters
"""

from scipy.spatial import distance_matrix
import numpy as np
import ast


def literal_eval(data, column_name):
    """
    apply ast's literal_eval to each entry in a column in a dataframe
    ast.literal_eval evaluates a string as list
    """

    literal_array = []
    for i in range(len(data)):

        literal_array.append(
            ast.literal_eval(data[column_name][i]))

    return np.array(literal_array)


def avg_distance_clusters(cluster1, cluster2):
    """
    compute avg distance between trajectrories of two clusters
    distance metric is L2 norm (euclidean distance) by default
    """

    pairwise_distances = distance_matrix(cluster1, cluster2)
    avg_distance = np.mean(pairwise_distances)

    return avg_distance


def avg_distance_all_clusters(data, labels, simulated_data):
    """
    find avg distance between simulated data and clusters from real data

    params:
        data (ndarray): trajectories from real data
        labels (list): from data specifying clusters
        simulated data (list): simulated trajectories from model

    returns:
        avg_distances (list): average distance to clusters
    """

    avg_distances = []
    for cluster in np.unique(labels):

        # select data from a cluster & find avg dist. to simulated trajectory
        avg_distances.append(
            avg_distance_clusters(data[np.where(labels == cluster)[0]],
                                  simulated_data))

    return avg_distances
