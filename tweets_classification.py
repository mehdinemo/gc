# Imports
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
import networkx as nx
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import svm
from config import config
from database_manager import DataBase
from tqdm import tqdm
from prepare_data import PrepareData


def scores_degree(G: nx.Graph) -> pd.DataFrame:
    print('creating degree of graph...')
    degrees_weight = G.degree(weight='jaccard_sim')
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
        sub_degrees_weight = v.degree(weight='jaccard_sim')
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


def scores_degree2(G: nx.Graph, method: str, sub_method: str, weight: str) -> pd.DataFrame:
    print(f'calculate {method} degree for graph')
    if method == 'degree':
        degrees = G.degree(weight=weight)
        degrees_df = pd.DataFrame(degrees, columns=['node', 'degree'])
    else:
        if method == 'closeness':
            degrees_df = nx.closeness_centrality(G, distance=weight)
        elif method == 'eig':
            degrees_df = nx.eigenvector_centrality(G, weight=weight)
        elif method == 'katz':
            degrees_df = nx.katz_centrality(G, weight=weight)
        degrees_df = pd.DataFrame.from_dict(degrees_df, orient='index')
        degrees_df.reset_index(inplace=True)
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
        if sub_method == 'degree':
            sub_deg = v.degree(weight=weight)
            sub_deg = pd.DataFrame(sub_deg, columns=['node', 'class_degree'])
        else:
            if sub_method == 'closeness':
                sub_deg = nx.closeness_centrality(v, distance=weight)
            elif sub_method == 'eig':
                sub_deg = nx.eigenvector_centrality(v, weight=weight)
            elif sub_method == 'katz':
                sub_deg = nx.katz_centrality(v, weight=weight)
            sub_deg = pd.DataFrame.from_dict(sub_deg, orient='index')
            sub_deg.reset_index(inplace=True)
            sub_deg.columns = ['node', 'class_degree']

        sub_deg_df = sub_deg_df.append(sub_deg)

    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')

    degrees_df['score'] = degrees_df['degree'] - degrees_df['class_degree']
    degrees_df.sort_values(by=['class', 'score'], ascending=False, inplace=True)

    degrees_df.drop(['degree', 'class_degree'], axis=1, inplace=True)

    return degrees_df


def scores_degree3(G: nx.Graph, weight: str) -> pd.DataFrame:
    print(f'calculate degree for graph')

    # degrees_df = nx.eigenvector_centrality(G, weight=weight)
    # degrees_df = pd.DataFrame.from_dict(degrees_df, orient='index')
    # degrees_df.reset_index(inplace=True)
    # degrees_df.columns = ['node', 'degree']
    # degrees_df.to_csv('data/degrees_df.csv', index=False)

    degrees_df = pd.read_csv('data/degrees_df.csv')
    # degrees_df.drop(degrees_df.columns.difference(['node', 'degree']), axis=1, inplace=True)
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
        # sub_deg = nx.eigenvector_centrality(v, max_iter=200, weight=weight)
        # sub_deg = pd.DataFrame.from_dict(sub_deg, orient='index')
        # sub_deg.reset_index(inplace=True)

        sub_deg = nx.degree(v, weight=weight)
        sub_deg = pd.DataFrame(sub_deg)

        sub_deg.columns = ['node', 'class_degree']
        # sub_deg['class_degree'] = sub_deg['class_degree'] / sub_deg['class_degree'].sum()

        sub_deg_df = sub_deg_df.append(sub_deg)

    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')
    degrees_df['degree'] = degrees_df['degree'] / degrees_df['degree'].sum()
    degrees_df['class_degree'] = degrees_df['class_degree'] / degrees_df['class_degree'].sum()

    degrees_df['score'] = degrees_df['class_degree'] - degrees_df['degree']
    # degrees_df['score'] = degrees_df['degree'] - degrees_df['class_degree']
    # degrees_df.sort_values(by=['class', 'score'], ascending=False, inplace=True)

    # degrees_df.to_csv('data/degrees_df.csv', index=False)
    degrees_df.drop(degrees_df.columns.difference(['node', 'class', 'score']), axis=1, inplace=True)

    return degrees_df


def fit_nodes(test_train_sim, scores, drop_index=False):
    test_train_sim = pd.DataFrame(test_train_sim)
    scores.set_index('node', inplace=True)

    # classes = scores['class'].unique()

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
        n_score = n_score.groupby(['class']).sum()
        n_score['score'] = n_score['score'] / n_score[index]
        n_score.drop([index], axis=1, inplace=True)

        n_score['score'] = n_score['score'].round(6)
        duplicated_labels = n_score['score'].duplicated(False)
        if (True in duplicated_labels.values) or (len(n_score) == 0):
            n_label = None
        else:
            n_label = n_score['score'].idxmax()
        predict.append(n_label)

    return predict


def fit_nodes2(test_train_sim, scores, n_select=0, drop_index=False):
    test_train_sim = pd.DataFrame(test_train_sim)
    scores.set_index(['node'], inplace=True)
    scores.sort_values(by=['class', 'score'], ascending=False, inplace=True)
    classes = scores['class'].unique()

    predict = []
    for index, row in tqdm(test_train_sim.iterrows(), total=test_train_sim.shape[0]):
        if drop_index:
            new_scores = scores.drop(index)
        else:
            new_scores = scores.copy()
        if n_select != 0:
            tops = pd.DataFrame(columns=scores.columns)
            botts = pd.DataFrame(columns=scores.columns)
            for c in classes:
                tops = tops.append(new_scores[new_scores['class'] == c].head(n_select))
                botts = botts.append(new_scores[new_scores['class'] == c].tail(n_select))

            top_ind = tops.merge(row, how='left', left_index=True, right_index=True)
            bott_ind = botts.merge(row, how='left', left_index=True, right_index=True)

            # top_ind = top_ind[top_ind[index] != 0]
            # bott_ind = bott_ind[bott_ind[index] != 0]
            #
            top_ind[index] = top_ind[index] * top_ind['score']
            bott_ind[index] = bott_ind[index] * bott_ind['score']

            top_ind = top_ind.groupby(['class'])[index].sum()
            bott_ind = bott_ind.groupby(['class'])[index].sum()

            n_score = 2 * bott_ind - top_ind
        else:
            row = row.to_frame()
            row = row.merge(new_scores, how='left', left_index=True, right_index=True)
            row = row[row[index] != 0]
            # row[index] = row[index] * row['score']
            n_score = row.groupby(['class'])[index].mean()

            # n_score = top_ind - bott_ind

        duplicated_labels = n_score.duplicated(False)
        if (True in duplicated_labels.values) or (len(n_score) == 0):
            n_label = None
        else:
            n_label = n_score.idxmax()
        predict.append(n_label)

    return predict


def scores_degree4(G: nx.Graph, weight: str, method: str, sub_method: str) -> pd.DataFrame:
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
    min_max_scaler = preprocessing.MinMaxScaler()
    sub_deg_df = pd.DataFrame()
    for k, v in subgraph_dic.items():
        print(f'calculate eigenvector_centrality for {k}')
        if sub_method == 'eig':
            sub_deg = nx.eigenvector_centrality(v, max_iter=200, weight=weight)
            sub_deg = pd.DataFrame.from_dict(sub_deg, orient='index')
            sub_deg.reset_index(inplace=True)
            sub_deg.columns = ['node', 'class_degree']
            # sub_deg['class_degree'] = min_max_scaler.fit_transform(sub_deg['class_degree'])
            # sub_deg['class_degree'] = sub_deg['class_degree'] / sub_deg['class_degree'].sum()
        elif sub_method == 'degree':
            sub_deg = nx.degree(v, weight=weight)
            sub_deg = pd.DataFrame(sub_deg)
            sub_deg.columns = ['node', 'class_degree']

        sub_deg_df = sub_deg_df.append(sub_deg)

    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')

    # degrees_df.to_csv('data/degrees_df.csv', index=False)

    # degrees_df['class_degree'] = degrees_df['class_degree'] / degrees_df['class_degree'].sum()
    # degrees_df['degree'] = degrees_df['degree'] / degrees_df['degree'].sum()
    # degrees_df['class_degree'] = degrees_df['class_degree'] / degrees_df['class_degree'].sum()

    # degrees_df['degree'] = min_max_scaler.fit_transform(degrees_df['degree'])

    # degrees_df['score'] = degrees_df['class_degree'] / degrees_df['degree']
    # degrees_df['score'] = degrees_df['class_degree'] - degrees_df['degree']
    # degrees_df['score'] = 2 * degrees_df['class_degree'] - degrees_df['degree']
    # degrees_df.sort_values(by=['class', 'score'], ascending=False, inplace=True)

    degrees_df.to_csv('data/degrees_df.csv', index=False)

    degrees_df.drop(degrees_df.columns.difference(['node', 'class', 'score']), axis=1, inplace=True)

    return degrees_df


def fit_nodes3(test_train_sim, train: pd.DataFrame):
    # scores.set_index(['node'], inplace=True)
    # # scores.sort_values(by=['class', 'score'], ascending=False, inplace=True)
    # classes = scores['class'].unique()

    train.set_index('id', inplace=True)
    test_train_sim = pd.DataFrame(test_train_sim)

    predict = []
    for index, row in tqdm(test_train_sim.iterrows(), total=test_train_sim.shape[0]):
        row = row.to_frame()
        row = row.merge(train, how='left', left_index=True, right_index=True)
        row = row[row[index] != 0]
        # row.sort_values(by=['class', 'score'], ascending=False, inplace=True)

        # tops = pd.DataFrame(columns=row.columns)
        # for c in classes:
        #     row_c = row[row['class'] == c]
        #     n = math.ceil(0.5 * len(row_c))
        #     tops = tops.append(row_c.head(n))
        #
        # # tops[index] = tops[index] * tops['score']
        # n_score = tops.groupby(['class'])[index].sum()

        # row[index] = row[index] * row['score']
        n_score = row.groupby(['target'])[index].sum()

        duplicated_labels = n_score.duplicated(False)
        if (True in duplicated_labels.values) or (len(n_score) == 0):
            n_label = None
        else:
            n_label = n_score.idxmax()
        predict.append(n_label)

    return predict


def main():
    pr = PrepareData()
    print('select data from db...')
    # data = db._select(query_string, connection_string)
    data = pd.read_csv(r'data/graph.csv')
    train = pd.read_csv(r'data/train.csv')
    train.rename(columns={'target': 'class'}, inplace=True)
    print('data loaded')

    data_sim = pr._jaccard_sim(data)

    clear_data_sim = pr._sim_nodes_detector(data_sim)

    print('creating graph...')
    G = nx.from_pandas_edgelist(clear_data_sim, source='source', target='target', edge_attr=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f'graph created with {len(G)} nodes and {G.number_of_edges()} edges.')

    node_dic = dict(zip(train['id'], train['class']))
    nx.set_node_attributes(G, node_dic, 'label')

    del data, data_sim, clear_data_sim

    # adjacency matrix
    sim = pr._adj_matrix(G, weight='jaccard_sim')

    # print('calculate scores...')
    scores = pd.DataFrame()
    # scores = pr._scores_degree(G, 'jaccard_sim', 'eig', 'eig')

    # # scores = scores_degree(G)
    # scores.to_csv(r'data/scores_tweet_eig.csv', index=False)
    # scores = pd.read_csv(r'data/scores_tweet_eig_degree.csv')
    # print('scores created')

    labels = nx.get_node_attributes(G, 'label')
    # n = math.ceil(0.15 * len(G))
    test_predict = pr._fit_nodes(sim, train[['id', 'class']], scores, 'max')
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

    # test_predict = test_predict.to_frame()
    # test_predict.index = sim.index
    # test_predict.to_csv('data/test_predict.csv')

    # # select test and train
    # data = train[train['id'].isin(all_nodes)].copy()
    # data.drop(data.columns.difference(['id', 'target']), 1, inplace=True)
    #
    # X_train, X_test, y_train, y_test = train_test_split(data['id'], data['target'], random_state=0)
    #
    # G_train = G.subgraph(X_train)
    # # G_test = G.subgraph(X_test)
    #
    # weight = 'jaccard_sim'
    #
    # print('calculate scores...')
    # scores_train = scores_degree4(G_train, weight)
    # # scores_train = pd.read_csv(r'data/scores_train.csv')
    #
    # # degrees_df = pd.read_csv(r'data/degrees_df.csv')
    # # degrees_df['degree'] = degrees_df['degree'] / degrees_df['degree'].max()
    # # # max_0 = degrees_df[degrees_df['class'] == 0]['class_degree'].max()
    # # # max_1 = degrees_df[degrees_df['class'] == 1]['class_degree'].max()
    # # # degrees_df['normal_class_degree'] = degrees_df.apply(
    # # #     lambda x: x['class_degree'] / max_0 if x['class'] == 0 else x['class_degree'] / max_1, axis=1)
    # # # degrees_df['score'] = degrees_df['degree'] - degrees_df['normal_class_degree']
    # # # degrees_df.drop(['normal_class_degree', 'degree', 'class_degree'], axis=1, inplace=True)
    # # # scores_train = degrees_df.copy()
    # #
    # # scores_train = scores_degree(G_train)
    # #
    # print('scores created')
    #
    # sim_test_train = sim.drop(X_train)
    # sim_test_train.drop(columns=X_test, axis=1, inplace=True)
    # # n = math.ceil(0.2 * len(G_train))
    # # test_predict = fit_nodes(sim_test_train, scores_train.copy(), n, False)
    # test_predict = fit_nodes3(sim_test_train, train[['id', 'target']], scores_train)
    # test_predict = pd.Series(test_predict).fillna(-1)
    # acc = classification_report(y_test, test_predict, output_dict=False)
    # print(acc)
    # #
    print('done')


if __name__ == '__main__':
    main()
