# Imports
import math
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
import networkx as nx
from sklearn.metrics import classification_report
from sklearn import svm
from tqdm import tqdm


def calculate_scores(G: nx.Graph, method: str, sub_method: str) -> pd.DataFrame:
    if method == 'degree':
        degrees = G.degree(weight='weight')
        degrees_df = pd.DataFrame(degrees, columns=['node', 'degree'])
    else:
        if method == 'closeness':
            degrees_df = nx.closeness_centrality(G, distance='weight')
        elif method == 'eig':
            degrees_df = nx.eigenvector_centrality(G, weight='weight')
        elif method == 'katz':
            degrees_df = nx.katz_centrality(G, weight='weight')
        degrees_df = pd.DataFrame.from_dict(degrees_df, orient='index')
        degrees_df.reset_index(inplace=True)
        degrees_df.columns = ['node', 'degree']

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
            sub_deg = v.degree(weight='weight')
            sub_deg = pd.DataFrame(sub_deg, columns=['node', 'class_degree'])
        else:
            if sub_method == 'closeness':
                sub_deg = nx.closeness_centrality(v, distance='weight')
            elif sub_method == 'eig':
                sub_deg = nx.eigenvector_centrality(v, weight='weight')
            elif sub_method == 'katz':
                sub_deg = nx.katz_centrality(v, weight='weight')
            sub_deg = pd.DataFrame.from_dict(sub_deg, orient='index')
            sub_deg.reset_index(inplace=True)
            sub_deg.columns = ['node', 'class_degree']

        sub_deg_df = sub_deg_df.append(sub_deg)

    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')

    degrees_df['score'] = degrees_df['degree'] - degrees_df['class_degree']
    degrees_df.sort_values(by=['class', 'score'], ascending=False, inplace=True)

    degrees_df.drop(['degree', 'class_degree'], axis=1, inplace=True)

    return degrees_df


def fit_nodes(test_train_sim, scores):
    test_train_sim = pd.DataFrame(test_train_sim)
    scores.drop(['node'], axis=1, inplace=True)
    predict = []
    for index, row in test_train_sim.iterrows():
        scores['score_sim'] = scores['score'] * row
        n_score = scores.groupby(by=['class'])['score_sim'].sum()

        duplicated_labels = n_score.duplicated(False)
        if True in duplicated_labels.values:
            n_label = None
        else:
            n_label = n_score.idxmax()
        predict.append(n_label)

    return predict


def fit_nodes2(test_train_sim, scores, n_select, drop_index):
    test_train_sim = pd.DataFrame(test_train_sim)
    scores.drop(['node'], axis=1, inplace=True)
    classes = scores['class'].unique()

    predict = []
    for index, row in test_train_sim.iterrows():
        if drop_index:
            new_scores = scores.drop(index)
        else:
            new_scores = scores.copy()
        tops = pd.DataFrame(columns=scores.columns)
        botts = pd.DataFrame(columns=scores.columns)
        for c in classes:
            tops = tops.append(new_scores[new_scores['class'] == c].head(n_select))
            botts = botts.append(new_scores[new_scores['class'] == c].tail(n_select))

        top_ind = tops.merge(row, how='left', left_index=True, right_index=True)
        bott_ind = botts.merge(row, how='left', left_index=True, right_index=True)

        top_ind = top_ind.groupby(['class'])[index].mean()
        bott_ind = bott_ind.groupby(['class'])[index].mean()

        n_score = 2 * bott_ind - top_ind

        duplicated_labels = n_score.duplicated(False)
        if True in duplicated_labels.values:
            n_label = None
        else:
            n_label = n_score.idxmax()
        predict.append(n_label)

    return predict


def prepare_data():
    # Dataset
    iris = datasets.load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)

    target = iris.target_names
    labels = iris.target

    # Scaling
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # PCA Transformation
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(data)
    PCAdf = pd.DataFrame(data=principalComponents,
                         columns=['principal component 1', 'principal component 2', 'principal component 3'])

    datapoints = PCAdf.values

    # svm
    # clf = svm.SVC(decision_function_shape='ovo')
    # clf.fit(datapoints, labels)
    # pred = clf.predict(datapoints)
    # acc = classification_report(labels, pred)
    # print(acc)

    dis = euclidean_distances(datapoints)
    sim = 1 / (1 + dis)
    G = nx.from_numpy_matrix(sim)
    G.remove_edges_from(nx.selfloop_edges(G))
    node_dic = dict(zip(range(0, len(datapoints)), labels))
    nx.set_node_attributes(G, node_dic, 'label')

    method = 'eig'
    sub_method = 'degree'
    # scores = calculate_scores(G, method, sub_method)

    # accuracy = pd.DataFrame()
    # n_range = [5, 10, 15, 20, 25]
    # for n in tqdm(n_range):
    #     train_predict = fit_nodes2(sim, scores.copy(), n, True)
    #     acc = classification_report(labels, train_predict, output_dict=True)
    #     # acc.update({'n': {'precision': n, 'recall': n, 'f1-score': n, 'support': n}})
    #     acc = pd.DataFrame(acc)
    #     acc = acc.T
    #     acc['n'] = n
    #     accuracy = accuracy.append(acc)
    #
    # accuracy.to_csv('data/accuracy.csv')

    # select test and train
    X_train, X_test, y_train, y_test = train_test_split(datapoints, labels, test_size=0.7)

    # distance and similarity
    dis_train = euclidean_distances(X_train)
    sim_train = 1 / (1 + dis_train)

    # build graph for train data
    G_train = nx.from_numpy_matrix(sim_train)
    G_train.remove_edges_from(nx.selfloop_edges(G_train))

    train_node_dic = dict(zip(range(0, len(X_train)), y_train))
    nx.set_node_attributes(G_train, train_node_dic, 'label')

    scores_train = calculate_scores(G_train, method, sub_method)

    test_train_dis = euclidean_distances(X_test, X_train)
    test_train_sim = 1 / (1 + test_train_dis)

    n = math.ceil(0.1 * len(X_train))
    test_predict = fit_nodes2(test_train_sim, scores_train.copy(), n, False)
    acc = classification_report(y_test, test_predict, output_dict=False)
    print(acc)

    # svm
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = classification_report(y_test, pred)
    print(acc)

    print('done')


def main():
    prepare_data()

    print('done')


if __name__ == '__main__':
    main()
