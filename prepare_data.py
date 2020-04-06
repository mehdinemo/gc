import pandas as pd
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import numpy as np


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


class PrepareData():
    def _jaccard_sim(self, data):
        nodes = data[data['source'] == data['target']].copy()
        nodes.drop(['target'], axis=1, inplace=True)
        nodes.columns = ['node', 'node_weigh']

        data = data.merge(nodes, how='left', left_on='source', right_on='node')
        data = data.merge(nodes, how='left', left_on='target', right_on='node')

        data['jaccard_sim'] = data['weight'] / (data['node_weigh_x'] + data['node_weigh_y'] - data['weight'])
        data.drop(data.columns.difference(['source', 'target', 'jaccard_sim']), axis=1, inplace=True)

        return data

    def _sim_nodes_detector(self, data_sim):
        sim_nodes = data_sim[data_sim['jaccard_sim'] == 1]
        sim_nodes = sim_nodes[sim_nodes['source'] != sim_nodes['target']]
        sim_nodes = sim_nodes.groupby('source')['target'].apply(list)

        sim_nodes = pd.DataFrame(sim_nodes)
        sim_nodes['is_similar'] = 0

        for row in sim_nodes.itertuples():
            if row.is_similar == 0:
                sim_nodes.at[row.target, 'is_similar'] = 1

        sim_nodes = sim_nodes[sim_nodes['is_similar'] == 1]

        data_sim = data_sim[(~data_sim['source'].isin(sim_nodes.index)) & (~data_sim['target'].isin(sim_nodes.index))]

        return data_sim

    def _scores_degree(self, G: nx.Graph, weight: str, method: str, sub_method: str) -> pd.DataFrame:
        print(f'calculate degree for graph')

        if method == 'eig':
            degrees_df = nx.eigenvector_centrality(G, weight=weight)
            degrees_df = pd.DataFrame.from_dict(degrees_df, orient='index')
            degrees_df.reset_index(inplace=True)
            degrees_df.columns = ['node', 'degree']
        elif method == 'degree':
            degrees_df = nx.degree(G, weight=weight)
            degrees_df = pd.DataFrame(degrees_df)
            degrees_df.columns = ['node', 'degree']

        print('graph degree calculated')

        classes = pd.DataFrame(nx.get_node_attributes(G, 'label').items(), columns=['node', 'class'])
        degrees_df = degrees_df.merge(classes, how='left', left_on='node', right_on='node')

        classes = classes['class'].unique()

        # create subgragps for each class
        subgraph_dic = {}
        for i in classes:
            sub_nodes = (
                node
                for node, data
                in G.nodes(data=True)
                if data.get('label') == i
            )
            subgraph = G.subgraph(sub_nodes)
            subgraph_dic.update({i: subgraph})

        # calculate degree for nodes in subgraphs
        # min_max_scaler = preprocessing.MinMaxScaler()
        sub_deg_df = pd.DataFrame()
        for k, v in subgraph_dic.items():
            print(f'calculate eigenvector_centrality for {k}')
            if sub_method == 'eig':
                sub_deg = nx.eigenvector_centrality(v, max_iter=200, weight=weight)
                sub_deg = pd.DataFrame.from_dict(sub_deg, orient='index')
                sub_deg.reset_index(inplace=True)
                sub_deg.columns = ['node', 'class_degree']
                # sub_deg['class_degree'] = min_max_scaler.fit_transform(sub_deg['class_degree'])
                sub_deg['class_degree'] = sub_deg['class_degree'] / sub_deg['class_degree'].sum()
            elif sub_method == 'degree':
                sub_deg = nx.degree(v, weight=weight)
                sub_deg = pd.DataFrame(sub_deg)
                sub_deg.columns = ['node', 'class_degree']

            sub_deg_df = sub_deg_df.append(sub_deg)

        degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')

        # degrees_df.to_csv('data/degrees_df.csv', index=False)

        degrees_df['class_degree'] = degrees_df['class_degree'] / degrees_df['class_degree'].sum()
        degrees_df['degree'] = degrees_df['degree'] / degrees_df['degree'].sum()
        # degrees_df['class_degree'] = degrees_df['class_degree'] / degrees_df['class_degree'].sum()

        # degrees_df['degree'] = min_max_scaler.fit_transform(degrees_df['degree'])

        # degrees_df['score'] = degrees_df['degree']
        degrees_df['score'] = degrees_df['class_degree'] / degrees_df['degree']
        # degrees_df['score'] = degrees_df['class_degree'] - degrees_df['degree']
        # degrees_df['score'] = 2 * degrees_df['class_degree'] - degrees_df['degree']
        # degrees_df.sort_values(by=['class', 'score'], ascending=False, inplace=True)

        # degrees_df.to_csv('data/degrees_df.csv', index=False)

        degrees_df.drop(degrees_df.columns.difference(['node', 'class', 'score']), axis=1, inplace=True)

        return degrees_df

    def _fit_nodes(self, sim, labels: pd.DataFrame, scores=pd.DataFrame(), nscore_method='sum'):
        if scores.empty:
            labels.set_index('id', inplace=True)
        else:
            scores.set_index('node', inplace=True)
        sim = pd.DataFrame(sim)

        predict = []
        for index, row in tqdm(sim.iterrows(), total=sim.shape[0]):
            row = row.to_frame()
            if scores.empty:
                row = row.merge(labels, how='left', left_index=True, right_index=True)
            else:
                row = row.merge(scores, how='left', left_index=True, right_index=True)
                row[index] = row[index] * row['score']

            row = row[row[index] != 0]

            if nscore_method == 'max':
                n_score = row.loc[row[index].idxmax()]
                n_label = n_score['class']
            else:
                if nscore_method == 'sum':
                    n_score = row.groupby(['class'])[index].sum()
                elif nscore_method == 'mean':
                    n_score = row.groupby(['class'])[index].sum()
                duplicated_labels = n_score.duplicated(False)
                if (True in duplicated_labels.values) or (len(n_score) == 0):
                    n_label = None
                else:
                    n_label = n_score.idxmax()
            predict.append(n_label)

        return predict

    def _adj_matrix(self, G: nx.Graph, weight='weight'):
        all_nodes = list(G.nodes)
        sim = nx.to_numpy_array(G, weight=weight)

        sim = pd.DataFrame(sim)
        sim.index = all_nodes
        sim.columns = all_nodes

        return sim

    def _load_data(self, path="data/cora/", dataset="cora"):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        samples = labels[np.random.choice(labels.shape[0], 140, replace=False)]

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize_features(features)
        # adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

        return features.todense(), adj, samples, labels
