import networkx.algorithms.community.centrality as nc
from sklearn.cluster import SpectralClustering
import scipy.spatial.distance as dis
import community
import networkx as nx
import numpy as np
import itertools
import pandas as pd
from sklearn.cluster import DBSCAN
import collections
from prepare_data import PrepareData


class ClusteringManipulator:
    def __init__(self):
        pass

    def clustering_matrix(self, G, sim=pd.DataFrame(), noise_deletion=True):
        pr = PrepareData()
        if sim.empty:
            sim = pr.adj_matrix(G)
        dis = 1 - sim

        if noise_deletion:
            dbs = DBSCAN(eps=0.9, min_samples=5, metric='precomputed').fit(dis)

            # Remove Noises from Graph
            noise_nodes = np.where(dbs.labels_ == -1)[0]
            noise_nodes = dis.index[noise_nodes]
            G.remove_nodes_from(noise_nodes)

        partitions = community.best_partition(G)

        if noise_deletion:
            noise_dic = {k: -1 for k in noise_nodes}
            partitions.update(noise_dic)
            # partitions = collections.OrderedDict(sorted(partitions.items()))

        partitions = pd.DataFrame.from_dict(partitions, orient='index')

        return partitions
