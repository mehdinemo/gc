import networkx as nx
import pandas as pd
from sklearn.metrics import classification_report
import time
from prepare_data import PrepareData
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg


def prepare_graph():
    cites = pd.read_csv(r'data\cites.csv', sep='\t', header=None)
    cites.columns = ['cited', 'citing']

    content = pd.read_csv(r'data\content.csv', sep='\t', header=None)
    # content.set_index(0, inplace=True)
    content.rename(columns={0: 'paper_id', content.columns[-1]: 'classes'}, inplace=True)
    classes = content[['paper_id', 'classes']].copy()
    content.drop(['classes'], axis=1, inplace=True)

    allkeywords = content.melt('paper_id', var_name='word', value_name='count')
    allkeywords = allkeywords[allkeywords['count'] != 0]
    allkeywords.drop(['count'], axis=1, inplace=True)

    graph = allkeywords.merge(allkeywords, how='inner', left_on='word', right_on='word')

    start_time = time.time()
    graph = graph.groupby(['paper_id_x', 'paper_id_y'], as_index=False)['word'].sum()
    end_time = time.time()

    print(end_time - start_time)

    graph.columns = ['source', 'target', 'weight']

    graph.to_csv('cora_graph.csv', index=False)


def prepare_data():
    pr = PrepareData()
    cites = pd.read_csv(r'data\cites.csv', sep='\t', header=None)
    cites.columns = ['source', 'target']
    G_cite = nx.from_pandas_edgelist(cites, source='source', target='target')

    # adjacency matrix
    sim_cite = pr._adj_matrix(G_cite)

    content = pd.read_csv(r'data\content.csv', sep='\t', header=None)
    # content.set_index(0, inplace=True)
    content.rename(columns={0: 'id', content.columns[-1]: 'class'}, inplace=True)
    classes = content[['id', 'class']].copy()
    # content.drop(['class'], axis=1, inplace=True)

    graph = pd.read_csv(r'data\cora_graph.csv')

    data_sim = pr._jaccard_sim(graph)
    clear_data_sim = pr._sim_nodes_detector(data_sim)

    print('creating graph...')
    G = nx.from_pandas_edgelist(clear_data_sim, source='source', target='target', edge_attr=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f'graph created with {len(G)} nodes and {G.number_of_edges()} edges.')

    # node_dic = dict(zip(classes['id'], classes['class']))
    # nx.set_node_attributes(G, node_dic, 'label')

    # L = nx.normalized_laplacian_matrix(G, weight='jaccard_sim')
    # e = numpy.linalg.eigvals(L.A)
    # print("Largest eigenvalue:", max(e))
    # print("Smallest eigenvalue:", min(e))
    # plt.hist(e, bins=100)  # histogram with 100 bins
    # plt.xlim(0, 2)  # eigenvalues between 0 and 2
    # plt.show()

    # adjacency matrix
    sim = pr._adj_matrix(G, 'jaccard_sim')

    # merge two sim
    # sim_cite[sim_cite == 0] = 0.5
    sim_multiply = sim + sim_cite
    sim_multiply.drop(sim_cite.columns.difference(sim.columns), axis=1, inplace=True)
    sim_multiply.dropna(axis=0, inplace=True)
    G = nx.from_pandas_adjacency(sim_multiply)

    node_dic = dict(zip(classes['id'], classes['class']))
    nx.set_node_attributes(G, node_dic, 'label')

    sim = pr._adj_matrix(G)

    print('calculate scores...')
    scores = pr._scores_degree(G, 'weight', 'eig', 'eig')

    labels = nx.get_node_attributes(G, 'label')
    # n = math.ceil(0.15 * len(G))
    test_predict = pr._fit_nodes(sim, classes[['id', 'class']], scores, 'max')
    test_predict = pd.Series(test_predict).fillna(-1)
    acc = classification_report(list(labels.values()), list(test_predict))
    print(acc)

    target = list(labels.values())
    test_predict = list(test_predict)
    true_predict = 0
    for i in range(len(target)):
        if target[i] == test_predict[i]:
            true_predict = true_predict + 1

    acc = true_predict / len(target)
    print(f'accuracy = {round(acc, 2)}')


def main():
    pr = PrepareData()
    # data = pd.read_csv(r'C:\Users\m.nemati\Desktop\nodes.csv')
    # g_data = data.groupby(['modularity_class', 'target'], as_index=False)['Id'].count()

    # features, adj, samples, labels = pr._load_data()

    # G = nx.from_scipy_sparse_matrix(adj)

    prepare_data()
    print('done')


if __name__ == '__main__':
    main()
