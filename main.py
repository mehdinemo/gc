import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from config import config

from database_manager import DataBase


def calculate_scores(G: nx.Graph) -> pd.DataFrame:
    degrees = G.degree(weight='weight')

    degrees_df = pd.DataFrame(degrees, columns=['node', 'degree'])
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
        sub_deg = v.degree(weight='weight')
        sub_deg_df = sub_deg_df.append(pd.DataFrame(sub_deg, columns=['node', 'class_degree']))

    # calculate scores, out of class degree for each nodes equals
    # to degree in main graph subtract from degree in subgraph
    # score equals to degree in subgraph subtract from degree in man graph
    # out = degree - class_degree, score = class_degree - out
    # score = class_degree - degree + class_degree = 2 class_degree - degree
    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')
    degrees_df['score'] = 2 * degrees_df['class_degree'] - degrees_df['degree']
    degrees_df.drop(['degree', 'class_degree'], axis=1, inplace=True)

    return degrees_df


def fit_nodes(scores_df: pd.DataFrame, test_edges: pd.DataFrame) -> pd.DataFrame:
    test_nodes_list = set(test_edges[~test_edges['source'].isin(scores_df['node'])]['source'])
    test_nodes_list.update(test_edges[~test_edges['target'].isin(scores_df['node'])]['target'])

    labels = {}
    for n in test_nodes_list:
        neighbours = test_edges[test_edges['source'] == n]['target']
        neighbours.append(test_edges[test_edges['target'] == n]['source'])

        neighbours_df = scores_df[scores_df['node'].isin(neighbours)]
        n_score = neighbours_df.groupby('class')['score'].sum()

        duplicated_labels = n_score.duplicated(False)
        if True in duplicated_labels.values:
            n_label = 0
        else:
            n_label = n_score.idxmax()
        labels.update({n: n_label})

    labels_df = pd.DataFrame(labels.items(), columns=['node', 'label'])

    return labels_df


def main():
    db = DataBase()
    connection_string = config['connection_string']

    with open(r'query/select_iris.sql', 'r')as file:
        query = file.read()

    data = db._select(query, connection_string)
    data['id'] = data['id'].astype(int)
    data['sepal_length'] = data['sepal_length'].astype(float)
    data['sepal_width'] = data['sepal_width'].astype(float)
    data['petal_length'] = data['petal_length'].astype(float)
    data['petal_width'] = data['petal_width'].astype(float)

    node_dict = dict(zip(data['id'], data['class']))

    similarities = cosine_similarity(data.drop(['id', 'class'], axis=1).values)
    sim_mat = np.matrix(similarities)

    G = nx.from_numpy_matrix(sim_mat)
    G.remove_edges_from(nx.selfloop_edges(G))

    # edges = pd.read_csv(r'data/edges.csv')
    # nodes = pd.read_csv(r'data/nodes.csv')
    # node_dict = dict(zip(nodes['node'], nodes['class']))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight')

    nx.set_node_attributes(G, node_dict, 'label')

    scores_df = calculate_scores(G)
    #
    # test_edges = pd.read_csv(r'data/test_edges.csv')
    # test_nodes = pd.read_csv(r'data/test_nodes.csv')
    #
    # test_labels = fit_nodes(scores_df, test_edges)
    #
    # test_labels = test_labels.merge(test_nodes, how='left', left_on='node', right_on='node')
    #
    # # nx.draw(subgraph)
    # # plt.show()

    print('done')


if __name__ == '__main__':
    main()
