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
from clustering_manipulator import ClusteringManipulator
import re

import nltk.corpus

nltk.download('stopwords')
from nltk.corpus import stopwords


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


def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df


def svm_toward(G: nx.Graph, train_data: pd.DataFrame, random_state=None, test_size=None):
    # train_data = train_data.drop(['keyword', 'location'], axis=1)

    data_clean = clean_text(train_data, "text")
    stop = stopwords.words('english')
    data_clean['text'] = data_clean['text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    labels = nx.get_node_attributes(G, 'label')
    labels = pd.DataFrame.from_dict(labels, orient='index')
    labels.reset_index(inplace=True)
    labels.columns = ['node', 'class']

    if G:
        X_train, X_test, y_train, y_test = train_test_split(labels['node'], labels['class'], random_state=random_state,
                                                            test_size=test_size)
        X_train = X_train.to_frame()
        X_train.columns = ['id']
        X_train = pd.merge(X_train, data_clean, 'left', left_on='id', right_on='id')
        y_train = X_train['class']
        X_train = X_train['text']

        X_test = X_test.to_frame()
        X_test.columns = ['id']
        X_test = pd.merge(X_test, data_clean, 'left', left_on='id', right_on='id')
        y_test = X_test['class']
        X_test = X_test['text']
    else:
        X_train, X_test, y_train, y_test = train_test_split(data_clean['text'], data_clean['class'], random_state=0)

    # from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import SGDClassifier
    pipeline_sgd = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('nb', SGDClassifier()),
    ])
    model = pipeline_sgd.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    print('SVM Results:')
    print(classification_report(y_test, y_predict))


def main():
    pr = PrepareData()
    cm = ClusteringManipulator()
    method = 'degree'
    sub_method = 'degree'
    # degree | eig
    delete_similar_data = True
    test = True
    # True | False
    label_method = 'max'
    # sum | mean | max
    random_state = 50
    # int | None
    n_head_score = 0.5
    # 0-1

    weight = 'jaccard_sim'

    print('select data from db...')
    # data = db._select(query_string, connection_string)
    data = pd.read_csv(r'data/graph_sample_fa.csv')
    train = pd.read_csv(r'data/sample_fa.csv')

    train = train[train['target'] == 1]

    data['source'] = data['source'].astype('int64')
    data['target'] = data['target'].astype('int64')

    data.drop(data.loc[data['source'] > data['target']].index.tolist(), inplace=True)

    train['id'] = train['id'].astype("Int64")

    train.rename(columns={'target': 'class'}, inplace=True)
    print('data loaded')

    data_sim = pr.jaccard_sim(data)
    if delete_similar_data:
        print('delete similar data...')
        data_sim, sim_dic = pr.sim_nodes_detector(data_sim)

    # data_sim['source'] = data_sim['source'].astype(str)
    # data_sim['target'] = data_sim['target'].astype(str)

    print('creating graph...')
    G = nx.from_pandas_edgelist(data_sim, source='source', target='target', edge_attr=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f'graph created with {len(G)} nodes and {G.number_of_edges()} edges.')

    node_dic = dict(zip(train['id'], train['class']))
    nx.set_node_attributes(G, node_dic, 'label')

    # prune graph with max weight
    sim = pr.adj_matrix(G, weight)
    print('prune graph...')
    G_p = pr.prune_max(sim)
    print('graph pruned!')
    # sim_p = pr._adj_matrix(G_p)
    # sim.to_csv('data/sim.csv')
    # sim_p.to_csv('data/sim_p.csv')

    partitions = cm.clustering_matrix(G_p, True)

    # longest_path = pr._longest_path(sim)
    # sim_lp = pr._adj_matrix(longest_path)
    # sim_lp.to_csv('data/sim_lp.csv')

    del data, data_sim

    if test:
        pr.test_graph(G, weight=weight, method=method, sub_method=sub_method, label_method=label_method,
                      random_state=random_state, n_head_score=n_head_score)

        # svm_toward(G, train, random_state=random_state)
        return

    # adjacency matrix
    sim = pr.adj_matrix(G, weight=weight)

    if method == '':
        scores = pd.DataFrame()
    else:
        print('calculate scores...')
        scores = pr.scores_degree(G, 'weight', method, sub_method)
        scores.to_csv('data/scores.csv', index=False)

    labels = nx.get_node_attributes(G, 'label')
    labels = pd.DataFrame.from_dict(labels, orient='index')
    labels.columns = ['class']

    # n = math.ceil(0.15 * len(G))
    test_predict = pr.fit_nodes(sim, labels, scores, label_method)
    pr.print_results(test_predict, labels)

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
