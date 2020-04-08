import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from prepare_data import PrepareData


# import numpy as np
# import matplotlib.pyplot as plt
# import numpy.linalg


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

    method = ''
    sub_method = ''
    # degree | eig

    delete_similar_data = False
    test = True
    # True | False

    label_method = 'max'
    # sum | mean | max

    random_state = 0
    # int |None

    cites = pd.read_csv(r'data\cites.csv', sep='\t', header=None)
    cites.columns = ['source', 'target']
    G_cite = nx.from_pandas_edgelist(cites, source='source', target='target')

    # adjacency matrix
    sim_cite = pr._adj_matrix(G_cite)

    content = pd.read_csv(r'data\content.csv', sep='\t', header=None)
    content.rename(columns={0: 'id', content.columns[-1]: 'class'}, inplace=True)
    classes = content[['id', 'class']].copy()

    graph = pd.read_csv(r'data\cora_graph.csv')

    data_sim = pr._jaccard_sim(graph)
    if delete_similar_data:
        print('delete similar data...')
        data_sim = pr._sim_nodes_detector(data_sim)

    print('creating graph...')
    G = nx.from_pandas_edgelist(data_sim, source='source', target='target', edge_attr=True)
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

    if test:
        pr._test_graph(G, label_method=label_method,random_state=random_state)
        return

    sim = pr._adj_matrix(G)

    if method == '':
        scores = pd.DataFrame()
    else:
        print('calculate scores...')
        scores = pr._scores_degree(G, 'weight', method, sub_method)

    labels = nx.get_node_attributes(G, 'label')
    labels = pd.DataFrame.from_dict(labels, orient='index')
    labels.columns = ['class']
    # n = math.ceil(0.15 * len(G))
    test_predict = pr._fit_nodes(sim, labels, scores, label_method)

    pr._print_results(test_predict, labels)


def main():
    pr = PrepareData()
    # data = pd.read_csv(r'C:\Users\m.nemati\Desktop\nodes.csv')
    # g_data = data.groupby(['modularity_class', 'target'], as_index=False)['Id'].count()

    # features, adj, samples, labels = pr._load_data()

    # G = nx.from_scipy_sparse_matrix(adj)

    prepare_data()
    print('done')


if __name__ == '__main__':
    # import sklearn
    # print(sklearn.__version__)
    main()
