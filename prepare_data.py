import pandas as pd
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import math


class PrepareData():
    def _jaccard_sim(self, data: pd.DataFrame) -> pd.DataFrame:
        nodes = data[data['source'] == data['target']].copy()
        nodes.drop(['target'], axis=1, inplace=True)
        nodes.columns = ['node', 'node_weigh']

        data = data.merge(nodes, how='left', left_on='source', right_on='node')
        data = data.merge(nodes, how='left', left_on='target', right_on='node')

        data['jaccard_sim'] = data['weight'] / (data['node_weigh_x'] + data['node_weigh_y'] - data['weight'])
        data.drop(data.columns.difference(['source', 'target', 'jaccard_sim']), axis=1, inplace=True)

        return data

    def _sim_nodes_detector(self, data_sim: pd.DataFrame) -> pd.DataFrame:
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

    def _longest_path(self, sim):
        sim_neg = -1 * sim
        G_neg = nx.from_pandas_adjacency(sim_neg)
        G_neg.remove_edges_from(nx.selfloop_edges(G_neg))

        lp = nx.shortest_path(G_neg)
        lp = next(iter(lp.values()))
        longest_past = list(lp.keys())

        G = nx.Graph()
        for i, v in enumerate(longest_past):
            if i + 1 < len(longest_past):
                target = longest_past[i + 1]
                weight = sim.loc[v][target]
                G.add_edge(v, target, weight=weight)
            else:
                target = longest_past[0]
                weight = sim.loc[v][target]
                G.add_edge(v, target, weight=weight)

        return G

    def _prune_max(self, sim: pd.DataFrame, weight='weight') -> pd.DataFrame:
        max_val = sim.max()
        max_ind = sim.idxmax()

        max_edges = pd.merge(max_ind.to_frame(), max_val.to_frame(), how='inner', left_index=True, right_index=True)
        max_edges.reset_index(inplace=True)
        max_edges.columns = ['source', 'target', weight]

        G_p = nx.from_pandas_edgelist(max_edges, source='source', target='target', edge_attr=True)
        cc = nx.connected_components(G_p)
        cc_list = []
        for c in cc:
            cc_list.append(c)

        check_nodes = cc_list.pop(0)
        while len(check_nodes) < len(G_p):
            tmp = sim.loc[check_nodes].drop(check_nodes, axis=1)
            max_ind = tmp.idxmax(axis=1)
            max_val = tmp.max(axis=1)
            source = max_val.idxmax()
            val = max_val.max()
            target = max_ind.loc[source]

            G_p.add_edge(source, target, weight=val)

            for i, v in enumerate(cc_list):
                if target in v:
                    break

            check_nodes = check_nodes | cc_list.pop(i)

        # print(
        #     f'graph created with {len(G_p)} nodes and {G_p.number_of_edges()} edges and {nx.number_connected_components(G_p)} connected components.')

        return G_p

    def _scores_degree(self, G: nx.Graph, weight='weight', method='degree', sub_method='degree') -> pd.DataFrame:
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
        sub_deg_df = pd.DataFrame()
        for k, v in subgraph_dic.items():
            print(f'calculate eigenvector_centrality for {k}')
            if sub_method == 'eig':
                sub_deg = nx.eigenvector_centrality(v, max_iter=200, weight=weight)
                sub_deg = pd.DataFrame.from_dict(sub_deg, orient='index')
                sub_deg.reset_index(inplace=True)
                sub_deg.columns = ['node', 'class_degree']
                sub_deg['class_degree'] = sub_deg['class_degree'] / sub_deg['class_degree'].sum()
            elif sub_method == 'degree':
                sub_deg = nx.degree(v, weight=weight)
                sub_deg = pd.DataFrame(sub_deg)
                sub_deg.columns = ['node', 'class_degree']

            sub_deg_df = sub_deg_df.append(sub_deg)

        degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')

        if method == 'eig':
            degrees_df['class_degree'] = degrees_df['class_degree'] / degrees_df['class_degree'].sum()
            degrees_df['degree'] = degrees_df['degree'] / degrees_df['degree'].sum()

        # degrees_df['score'] = degrees_df['degree']
        degrees_df['score'] = degrees_df['class_degree'] / degrees_df['degree']
        # degrees_df['score'] = degrees_df['class_degree'] - degrees_df['degree']
        # degrees_df['score'] = 2 * degrees_df['class_degree'] - degrees_df['degree']

        degrees_df.drop(degrees_df.columns.difference(['node', 'class', 'score']), axis=1, inplace=True)

        return degrees_df

    def _fit_nodes(self, sim, labels: pd.DataFrame, scores=pd.DataFrame(), nscore_method='max') -> pd.DataFrame:
        if not scores.empty:
            scores.set_index('node', inplace=True)
        sim = pd.DataFrame(sim)

        # max_edges = pd.DataFrame(columns=['source', 'target', 'weight'])
        predict = pd.DataFrame(columns=['node', 'label'])
        for index, row in tqdm(sim.iterrows(), total=sim.shape[0]):
            row = row.to_frame()
            if scores.empty:
                row = row.merge(labels, how='left', left_index=True, right_index=True)
            else:
                row = row.merge(scores, how='left', left_index=True, right_index=True)
                # row[index] = row[index] * row['score']

            row.dropna(inplace=True)
            # row = row[row[index] != 0]

            if nscore_method == 'max':
                if len(row) > 0:
                    try:
                        ind_max = row[index].idxmax()
                        n_score = row.loc[ind_max]
                        n_label = n_score['class']
                        # max_edges = max_edges.append({'source': index, 'target': ind_max, 'weight': n_score[index]},
                        #                              ignore_index=True)
                    except Exception as ex:
                        print(ex)
                else:
                    n_label = None
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
            predict = predict.append({'node': index, 'label': n_label}, ignore_index=True)

        predict.set_index('node', inplace=True)

        # predict.to_csv('data/predict.csv')
        # max_edges.to_csv('data/max_edges.csv', index=False)

        return predict

    def _adj_matrix(self, G: nx.Graph, weight='weight'):
        all_nodes = list(G.nodes)
        sim = nx.to_numpy_array(G, weight=weight)

        sim = pd.DataFrame(sim)
        sim.index = all_nodes
        sim.columns = all_nodes

        return sim

    def _print_results(self, test_predict: pd.DataFrame, labels: pd.DataFrame):
        test_predict.fillna(-1, inplace=True)
        test_predict = pd.merge(test_predict, labels, how='left', left_index=True, right_index=True)
        test_predict.to_csv('data/all_sample_predict.csv')
        acc = classification_report(test_predict['class'], test_predict['label'], output_dict=False)
        print(acc)

    def _test_graph(self, G: nx.Graph, weight='weight', method='', sub_method='', test_size=None, random_state=None,
                    label_method='max', n_head_score=1):
        pr = PrepareData()

        labels = nx.get_node_attributes(G, 'label')
        labels = pd.DataFrame.from_dict(labels, orient='index')
        labels.reset_index(inplace=True)
        labels.columns = ['node', 'class']

        X_train, X_test, y_train, y_test = train_test_split(labels['node'], labels['class'], random_state=random_state,
                                                            test_size=test_size)
        G_train = G.subgraph(X_train)

        if method == '':
            scores_train = pd.DataFrame()
        else:
            print('calculate scores...')
            scores_train = pr._scores_degree(G_train, weight, method=method, sub_method=sub_method)

            scores_train.sort_values(by=['class', 'score'], ascending=False, inplace=True)
            classes = scores_train['class'].unique()
            scores_sorted = pd.DataFrame()
            for c in classes:
                c_score = scores_train[scores_train['class'] == c].copy()
                c_score.sort_values(by=['class', 'score'], ascending=False, inplace=True)
                n = math.ceil(n_head_score * len(c_score))
                c_score = c_score.head(n)
                scores_sorted = scores_sorted.append(c_score)

            scores_train = scores_sorted
            print('scores created')

        labels.set_index('node', inplace=True)
        # adjacency matrix
        sim = pr._adj_matrix(G, weight)

        sim_test_train = sim.drop(list(X_train.values))
        sim_test_train.drop(columns=list(X_test.values), axis=1, inplace=True)
        test_predict = pr._fit_nodes(sim_test_train, labels, scores_train, label_method)

        pr._print_results(test_predict, labels)

    def _encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    def _load_data(self, path="data/cora/", dataset="cora"):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self._encode_onehot(idx_features_labels[:, -1])
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
