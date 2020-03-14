# Imports
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
import networkx as nx


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
    # m, f = datapoints.shape
    # k = 3

    # select test and train
    X_train, X_test, y_train, y_test = train_test_split(datapoints, labels, test_size=0.25)

    # distance and similarity
    dis = euclidean_distances(X_train)
    sim = 1 / (1 + dis)

    # build graph for train data
    G = nx.from_numpy_matrix(sim)
    G.remove_edges_from(nx.selfloop_edges(G))

    node_dic = dict(zip(range(0, len(X_train)), y_train))
    nx.set_node_attributes(G, node_dic, 'label')

    scores_df = calculate_scores(G)

    print('done')


def main():
    prepare_data()

    print('done')


if __name__ == '__main__':
    main()
