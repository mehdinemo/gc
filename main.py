import networkx as nx
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_scores(G: nx.Graph) -> pd.DataFrame:
    degrees_df = nx.eigenvector_centrality(G, weight='weight')
    degrees_df = pd.DataFrame.from_dict(degrees_df, orient='index')
    degrees_df.reset_index(inplace=True)
    degrees_df.columns = ['node', 'degree']

    classes = pd.DataFrame(nx.get_node_attributes(G, 'label').items(), columns=['node', 'class'])
    degrees_df = degrees_df.merge(classes, how='left', left_on='node', right_on='node')

    classes = classes['class'].unique()

    # create generator
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

    sub_deg_df = pd.DataFrame()
    for k, v in subgraph_dic.items():
        sub_deg = nx.eigenvector_centrality(v, weight='weight')
        sub_deg = pd.DataFrame.from_dict(sub_deg, orient='index')
        sub_deg.reset_index(inplace=True)
        sub_deg.columns = ['node', 'class_degree']
        sub_deg['class_degree'] = sub_deg['class_degree'] / sub_deg['class_degree'].sum()

        sub_deg_df = sub_deg_df.append(sub_deg)

    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')
    degrees_df['degree'] = degrees_df['degree'] / degrees_df['degree'].sum()
    degrees_df['class_degree'] = degrees_df['class_degree'] / degrees_df['class_degree'].sum()

    degrees_df['score'] = degrees_df['class_degree'] - degrees_df['degree']
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
        # n_score = n_score.groupby(['class']).agg({'score': 'sum', index: 'sum'})
        n_score = n_score.groupby(['class']).sum()
        n_score['score'] = n_score['score'] / n_score[index]
        n_score.drop([index], axis=1, inplace=True)

        n_score['score'] = n_score['score'].round(5)
        duplicated_labels = n_score['score'].duplicated(False)
        if (True in duplicated_labels.values) or (len(n_score) == 0):
            n_label = None
        else:
            n_label = n_score['score'].idxmax()
        predict.append(n_label)

    return predict


def main():
    edges = pd.read_csv(r'data/edges.csv')
    nodes = pd.read_csv(r'data/nodes.csv')
    node_dict = dict(zip(nodes['node'], nodes['class']))

    G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight')
    nx.set_node_attributes(G, node_dict, 'label')

    # bet = nx.eigenvector_centrality(G, weight='weight')
    # bet = pd.DataFrame.from_dict(bet, orient='index')
    # bet.reset_index(inplace=True)
    # bet.columns = ['node', 'degree']
    # bet['degree'] = bet['degree'] / bet['degree'].sum()

    # sim = nx.to_numpy_array(G, weight='weight')

    scores_df = calculate_scores(G)

    test_edges = pd.read_csv(r'data/test_edges.csv')
    test_nodes = pd.read_csv(r'data/test_nodes.csv')

    G_test = nx.from_pandas_edgelist(test_edges, source='source', target='target', edge_attr='weight')

    all_nodes = list(G_test.nodes)

    sim_test = nx.to_numpy_array(G_test, weight='weight')

    sim_test = pd.DataFrame(sim_test)
    sim_test.index = all_nodes
    sim_test.columns = all_nodes
    sim_test.drop(G.nodes, inplace=True)
    sim_test.drop(columns=all_nodes - G.nodes, axis=1, inplace=True)

    test_labels = fit_nodes(sim_test, scores_df)

    test_labels = test_labels.merge(test_nodes, how='left', left_on='node', right_on='node')

    # nx.draw(subgraph)
    # plt.show()

    print('done')


if __name__ == '__main__':
    main()
