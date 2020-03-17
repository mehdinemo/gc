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


def load_iris():
    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

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

    results = {'data': datapoints, 'labels': labels}
    return results


def create_graph(datapoints, labels) -> nx.Graph:
    dis = euclidean_distances(datapoints)
    sim = 1 / (1 + dis)
    G = nx.from_numpy_matrix(sim)
    G.remove_edges_from(nx.selfloop_edges(G))
    node_dic = dict(zip(range(0, len(datapoints)), labels))
    nx.set_node_attributes(G, node_dic, 'label')

    return G


def calculate_scores(G: nx.Graph, method: str, sub_method: str) -> pd.DataFrame:
    if method == 'degree':
        print('creating degree of graph...')
        degrees = G.degree(weight='weight')
        degrees_df = pd.DataFrame(degrees, columns=['node', 'degree'])
        print('degree created')
    else:
        if method == 'closeness':
            degrees_df = nx.closeness_centrality(G, distance='weight')
        elif method == 'eig':
            print('crating eigenvector_centrality for graph...')
            degrees_df = nx.eigenvector_centrality(G, weight='weight')
            print('eigenvector_centrality created')
        elif method == 'katz':
            degrees_df = nx.katz_centrality(G, weight='weight')
        degrees_df = pd.DataFrame.from_dict(degrees_df, orient='index')
        degrees_df.reset_index(inplace=True)
        degrees_df.columns = ['node', 'degree']

    classes = pd.DataFrame(nx.get_node_attributes(G, 'label').items(), columns=['node', 'class'])
    degrees_df = degrees_df.merge(classes, how='left', left_on='node', right_on='node')

    classes = classes['class'].unique()

    # create subgragps for each class
    print('create subgragps for each class...')
    subgraph_dic = {}
    for i in tqdm(classes):
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

    degrees_df.to_csv(r'data/degrees_df.csv')
    degrees_df.drop(['degree', 'class_degree'], axis=1, inplace=True)

    return degrees_df


def fit_nodes(test_train_sim, scores, n_select, drop_index):
    test_train_sim = pd.DataFrame(test_train_sim)
    scores.set_index('node', inplace=True)
    # scores.drop(['node'], axis=1, inplace=True)
    classes = scores['class'].unique()

    predict = []
    print('fit nodes...')
    for index, row in tqdm(test_train_sim.iterrows(), total=test_train_sim.shape[0]):
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

        top_ind['score'] = top_ind['score'] * top_ind[index]
        bott_ind['score'] = bott_ind['score'] * bott_ind[index]

        top_ind = top_ind[top_ind['score'] != 0]
        bott_ind = bott_ind[bott_ind['score'] != 0]

        top_ind = top_ind.groupby(['class'])['score'].mean()
        bott_ind = bott_ind.groupby(['class'])['score'].mean()

        n_score = 2 * bott_ind - top_ind

        duplicated_labels = n_score.duplicated(False)
        if True in duplicated_labels.values:
            n_label = None
        else:
            n_label = n_score.idxmax()
        predict.append(n_label)

    return predict


def iris_classification():
    res = load_iris()
    datapoints = res['data']
    labels = res['labels']

    # svm
    # clf = svm.SVC(decision_function_shape='ovo')
    # clf.fit(datapoints, labels)
    # pred = clf.predict(datapoints)
    # acc = classification_report(labels, pred)
    # print(acc)

    G = create_graph(datapoints, labels)
    sim = nx.to_numpy_array(G)

    method = 'eig'
    sub_method = 'degree'
    scores = calculate_scores(G, method, sub_method)

    accuracy = pd.DataFrame()
    n_range = [5, 10, 15, 20, 25]
    for n in tqdm(n_range):
        train_predict = fit_nodes(sim, scores.copy(), n, True)
        acc = classification_report(labels, train_predict, output_dict=True)
        acc = pd.DataFrame(acc)
        acc = acc.T
        acc['n'] = n
        accuracy = accuracy.append(acc)

    # accuracy.to_csv('data/accuracy.csv')

    # select test and train
    X_train, X_test, y_train, y_test = train_test_split(datapoints, labels, test_size=0.7)

    # build graph for train data
    G_train = create_graph(X_train, y_train)

    scores_train = calculate_scores(G_train, method, sub_method)

    test_train_dis = euclidean_distances(X_test, X_train)
    test_train_sim = 1 / (1 + test_train_dis)

    n = math.ceil(0.1 * len(X_train))
    test_predict = fit_nodes(test_train_sim, scores_train.copy(), n, False)
    acc = classification_report(y_test, test_predict, output_dict=False)
    print(acc)


def scores_degree(G: nx.Graph) -> pd.DataFrame:
    print('creating degree of graph...')
    degrees = G.degree(weight='weight')
    degrees_df = pd.DataFrame(degrees, columns=['node', 'degree'])
    print('degree created')

    classes = pd.DataFrame(nx.get_node_attributes(G, 'label').items(), columns=['node', 'class'])
    degrees_df = degrees_df.merge(classes, how='left', left_on='node', right_on='node')

    classes = classes['class'].unique()

    # create subgragps for each class
    print('create subgragps for each class...')
    subgraph_dic = {}
    for i in tqdm(classes):
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
        sub_deg = v.degree(weight='weight')
        sub_deg = pd.DataFrame(sub_deg, columns=['node', 'class_degree'])
        sub_deg_df = sub_deg_df.append(sub_deg)

    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')

    degrees_df['score'] = 2 * degrees_df['class_degree'] - degrees_df['degree']
    degrees_df.sort_values(by=['class', 'score'], ascending=False, inplace=True)

    degrees_df.drop(['degree', 'class_degree'], axis=1, inplace=True)

    return degrees_df


def fit_nodes2(test_train_sim, scores, n_select, drop_index):
    test_train_sim = pd.DataFrame(test_train_sim)
    scores.set_index('node', inplace=True)
    # scores.drop(['node'], axis=1, inplace=True)
    classes = scores['class'].unique()

    predict = []
    print('fit nodes...')
    for index, row in tqdm(test_train_sim.iterrows(), total=test_train_sim.shape[0]):
        if drop_index:
            new_scores = scores.drop(index)
        else:
            new_scores = scores.copy()

        top_ind = new_scores.merge(row, how='left', left_index=True, right_index=True)
        # bott_ind = new_scores.merge(row, how='left', left_index=True, right_index=True)

        top_ind['score'] = top_ind['score'] * top_ind[index]
        top_ind = top_ind.groupby(['class'])['score'].sum()
        # bott_ind = bott_ind.groupby(['class'])[index].mean()

        # n_score = 2 * bott_ind - top_ind

        n_score = top_ind
        duplicated_labels = n_score.duplicated(False)
        if True in duplicated_labels.values:
            n_label = None
        else:
            n_label = n_score.idxmax()
        predict.append(n_label)

    return predict


def main():
    # iris_classification()

    connection_string = config['connection_string']
    db = DataBase()
    with open(r'query/tweet_graph.sql')as file:
        query_string = file.read()

    print('select data from db...')
    # data = db._select(query_string, connection_string)
    data = pd.read_csv(r'data/graph.csv')
    print('data loaded')

    print('creating graph...')
    G = nx.from_pandas_edgelist(data, source='source', target='target', edge_attr='weight')
    print('graph created')

    del data
    G.remove_edges_from(nx.selfloop_edges(G))

    train = pd.read_csv(r'data/train.csv')

    node_dic = dict(zip(train['id'], train['target']))
    nx.set_node_attributes(G, node_dic, 'label')

    all_nodes = list(G.nodes)

    sim = nx.to_numpy_array(G)

    sim = pd.DataFrame(sim)
    sim.index = all_nodes
    sim.columns = all_nodes

    # method = 'closeness'
    # sub_method = 'degree'
    # print('calculate scores...')
    # # scores = calculate_scores(G, method, sub_method)
    # scores = scores_degree(G)
    # print('scores created')
    #
    # labels = nx.get_node_attributes(G, 'label')
    # n = math.ceil(0.1 * len(G))
    # test_predict = fit_nodes2(sim, scores, n, True)
    # test_predict = pd.Series(test_predict).fillna(-1)
    # acc = classification_report(list(labels.values()), test_predict, output_dict=False)
    # print(acc)

    # select test and train
    data = train[train['id'].isin(all_nodes)]
    data.drop(data.columns.difference(['id', 'target']), 1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data['id'], data['target'], random_state=0)

    G_train = G.subgraph(X_train)
    # G_test = G.subgraph(X_test)

    method = 'eig'
    sub_method = 'degree'

    print('calculate scores...')
    scores_train = calculate_scores(G_train, method, sub_method)
    # scores_train = pd.read_csv(r'data/scores_train.csv')
    # scores_train = scores_degree(G_train)
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
