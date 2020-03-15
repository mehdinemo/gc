# Imports
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
import networkx as nx
from sklearn.metrics import classification_report


def calculate_scores(G: nx.Graph) -> pd.DataFrame:
    # degree
    # degrees = G.degree(weight='weight')
    # degrees_df = pd.DataFrame(degrees, columns=['node', 'degree'])

    # closeness
    closeness = nx.closeness_centrality(G, distance='weight')
    degrees_df = pd.DataFrame.from_dict(closeness, orient='index')
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
        # degree
        # sub_deg = v.degree(weight='weight')
        # sub_deg = pd.DataFrame(sub_deg, columns=['node', 'class_degree'])

        # closeness
        sub_deg = nx.closeness_centrality(v, distance='weight')
        sub_deg = pd.DataFrame.from_dict(sub_deg, orient='index')
        sub_deg.reset_index(inplace=True)
        sub_deg.columns = ['node', 'class_degree']

        sub_deg_df = sub_deg_df.append(sub_deg)

    # calculate scores, out of class degree for each nodes equals
    # to degree in main graph subtract from degree in subgraph
    # score equals to degree in subgraph subtract from degree in man graph
    # out = degree - class_degree, score = class_degree - out
    # score = class_degree - degree + class_degree = 2 class_degree - degree
    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')

    # degree
    # degrees_df['score'] = 2 * degrees_df['class_degree'] - degrees_df['degree']

    # closeness
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


def fit_nodes2(test_train_sim, scores, n_select):
    test_train_sim = pd.DataFrame(test_train_sim)
    scores.drop(['node'], axis=1, inplace=True)
    classes = scores['class'].unique()
    tops = pd.DataFrame(columns=scores.columns)
    botts = pd.DataFrame(columns=scores.columns)
    for c in classes:
        tops = tops.append(scores[scores['class'] == c].head(n_select))
        botts = botts.append(scores[scores['class'] == c].tail(n_select))

    predict = []
    for index, row in test_train_sim.iterrows():
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

    dis = euclidean_distances(datapoints)
    sim = 1 / (1 + dis)
    G = nx.from_numpy_matrix(sim)
    G.remove_edges_from(nx.selfloop_edges(G))
    node_dic = dict(zip(range(0, len(datapoints)), labels))
    nx.set_node_attributes(G, node_dic, 'label')

    scores = calculate_scores(G)
    train_predict = fit_nodes2(sim, scores, 5)

    acc = classification_report(labels, train_predict)
    print(acc)
    # select test and train
    X_train, X_test, y_train, y_test = train_test_split(datapoints, labels, test_size=0.25)

    # distance and similarity
    dis_train = euclidean_distances(X_train)
    sim_train = 1 / (1 + dis_train)

    # build graph for train data
    G_train = nx.from_numpy_matrix(sim_train)
    G_train.remove_edges_from(nx.selfloop_edges(G_train))

    train_node_dic = dict(zip(range(0, len(X_train)), y_train))
    nx.set_node_attributes(G_train, train_node_dic, 'label')

    scores_train = calculate_scores(G_train)

    test_train_dis = euclidean_distances(X_test, X_train)
    test_train_sim = 1 / (1 + test_train_dis)

    test_predict = fit_nodes(test_train_sim, scores_train)

    print('done')


def main():
    prepare_data()

    print('done')


if __name__ == '__main__':
    main()
