# Imports
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
import networkx as nx
from sklearn.metrics import classification_report
from sklearn import svm
from config import config
from database_manager import DataBase
from tqdm import tqdm


def scores_degree(G: nx.Graph) -> pd.DataFrame:
    print('creating degree of graph...')
    degrees_weight = G.degree(weight='weight')
    # degrees_conf = G.degree(weight='confidence')
    # degrees_degree = G.degree()
    degrees_df = pd.DataFrame(degrees_weight, columns=['node', 'weight'])
    # degrees_conf = pd.DataFrame(degrees_conf, columns=['node', 'confidence'])
    # degrees_degree = pd.DataFrame(degrees_degree, columns=['node', 'degree'])
    # degrees_df = degrees_df.merge(degrees_degree, how='inner', left_on='node', right_on='node')
    # degrees_df = degrees_df.merge(degrees_conf, how='inner', left_on='node', right_on='node')

    print('degree created')

    classes = pd.DataFrame(nx.get_node_attributes(G, 'label').items(), columns=['node', 'class'])
    degrees_df = degrees_df.merge(classes, how='left', left_on='node', right_on='node')

    classes = classes['class'].unique()

    # create subgragps for each class
    print('create subgragps for each class...')
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
    print('calculate degree for nodes in subgraphs...')
    sub_deg_df = pd.DataFrame()
    for k, v in tqdm(subgraph_dic.items(), total=len(subgraph_dic)):
        sub_degrees_weight = v.degree(weight='weight')
        # sub_degrees_conf = v.degree(weight='confidence')
        # sub_degrees_degree = v.degree()
        sub_deg = pd.DataFrame(sub_degrees_weight, columns=['node', 'sub_weight'])
        # sub_degrees_conf = pd.DataFrame(sub_degrees_conf, columns=['node', 'sub_confidence'])
        # sub_degrees_degree = pd.DataFrame(sub_degrees_degree, columns=['node', 'sub_degree'])

        # sub_deg = sub_degrees_degree.merge(sub_deg, how='inner', left_on='node', right_on='node')
        # sub_deg = sub_deg.merge(sub_degrees_conf, how='inner', left_on='node', right_on='node')

        sub_deg_df = sub_deg_df.append(sub_deg)

    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')
    degrees_df['score'] = 2 * degrees_df['sub_weight'] - degrees_df['weight']

    # degrees_df = degrees_df.apply(change_confidence, axis=1)

    degrees_df.sort_values(by=['class', 'score'], ascending=False, inplace=True)
    degrees_df.drop(degrees_df.columns.difference(['node', 'class', 'score']), axis=1, inplace=True)
    # degrees_df.to_csv(r'data/degrees_df.csv', index=False)
    return degrees_df


def fit_nodes(test_train_sim, scores, n_select, drop_index):
    test_train_sim = pd.DataFrame(test_train_sim)
    scores.set_index('node', inplace=True)

    classes = scores['class'].unique()

    predict = []
    print('fit nodes...')
    for index, row in tqdm(test_train_sim.iterrows(), total=test_train_sim.shape[0]):
        if drop_index:
            new_scores = scores.drop(index)
        else:
            new_scores = scores.copy()

        n_score = new_scores.merge(row, how='left', left_index=True, right_index=True)

        n_score['score'] = n_score['score'] * n_score[index]
        n_score.fillna(0, inplace=True)
        n_score = n_score[n_score['score'] != 0]
        # n_score = n_score.groupby(['class']).agg({'score': 'sum', index: 'sum'})
        n_score = n_score.groupby(['class']).sum()
        n_score['score'] = n_score['score'] / n_score[index]
        n_score.drop([index], axis=1, inplace=True)

        duplicated_labels = n_score['score'].duplicated(False)
        if (True in duplicated_labels.values) or (len(n_score) == 0):
            n_label = None
        else:
            n_label = n_score['score'].idxmax()
        predict.append(n_label)

    return predict


def main():
    # iris_classification()

    # connection_string = config['connection_string']
    # db = DataBase()
    # with open(r'query/tweet_graph.sql')as file:
    #     query_string = file.read()

    print('select data from db...')
    # data = db._select(query_string, connection_string)
    data = pd.read_csv(r'data/graph.csv')
    print('data loaded')

    print('creating graph...')
    G = nx.from_pandas_edgelist(data, source='source', target='target', edge_attr=True)
    del data
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f'graph created with {len(G)} nodes and {G.number_of_edges()} edges.')

    train = pd.read_csv(r'data/train.csv')

    node_dic = dict(zip(train['id'], train['target']))
    nx.set_node_attributes(G, node_dic, 'label')

    all_nodes = list(G.nodes)

    sim = nx.to_numpy_array(G)

    sim = pd.DataFrame(sim)
    sim.index = all_nodes
    sim.columns = all_nodes
    # sim.to_csv(r'data/sim_all.csv')

    # method = 'closeness'
    # sub_method = 'degree'
    print('calculate scores...')
    # scores = calculate_scores(G, method, sub_method)
    scores = scores_degree(G)
    print('scores created')

    # labels = nx.get_node_attributes(G, 'label')
    # n = math.ceil(0.1 * len(G))
    # test_predict = fit_nodes(sim, scores, n, True)
    # test_predict = pd.Series(test_predict).fillna(-1)
    # acc = classification_report(list(labels.values()), test_predict, output_dict=False)
    # print(acc)

    # select test and train
    data = train[train['id'].isin(all_nodes)].copy()
    data.drop(data.columns.difference(['id', 'target']), 1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data['id'], data['target'], random_state=0)

    G_train = G.subgraph(X_train)
    # G_test = G.subgraph(X_test)

    method = 'eig'
    sub_method = 'degree'

    print('calculate scores...')
    # scores_train = calculate_scores(G_train, method, sub_method)
    # scores_train = pd.read_csv(r'data/scores_train.csv')

    # degrees_df = pd.read_csv(r'data/degrees_df.csv')
    # degrees_df['degree'] = degrees_df['degree'] / degrees_df['degree'].max()
    # max_0 = degrees_df[degrees_df['class'] == 0]['class_degree'].max()
    # max_1 = degrees_df[degrees_df['class'] == 1]['class_degree'].max()
    # degrees_df['normal_class_degree'] = degrees_df.apply(
    #     lambda x: x['class_degree'] / max_0 if x['class'] == 0 else x['class_degree'] / max_1, axis=1)
    # degrees_df['score'] = degrees_df['degree'] - degrees_df['normal_class_degree']
    # degrees_df.drop(['normal_class_degree', 'degree', 'class_degree'], axis=1, inplace=True)
    # scores_train = degrees_df.copy()

    scores_train = scores_degree(G_train)

    print('scores created')

    sim_test_train = sim.drop(X_train)
    sim_test_train.drop(columns=X_test, axis=1, inplace=True)
    n = math.ceil(0.2 * len(G_train))
    test_predict = fit_nodes(sim_test_train, scores_train.copy(), n, False)
    test_predict = pd.Series(test_predict).fillna(-1)
    acc = classification_report(y_test, test_predict, output_dict=False)
    print(acc)

    print('done')


if __name__ == '__main__':
    main()
