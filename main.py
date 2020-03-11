import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def calculate_scores(G: nx.Graph) -> pd.DataFrame:
    degrees = G.degree()

    degrees_df = pd.DataFrame(degrees, columns=['node', 'degree'])
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
        sub_deg = v.degree()
        sub_deg_df = sub_deg_df.append(pd.DataFrame(sub_deg, columns=['node', 'class_degree']))

    degrees_df = degrees_df.merge(sub_deg_df, how='left', left_on='node', right_on='node')
    degrees_df['score'] = 2 * degrees_df['class_degree'] - degrees_df['degree']
    degrees_df.drop(['degree', 'class_degree'], axis=1, inplace=True)

    return degrees_df


def fit_nodes(scores_df: pd.DataFrame, test_edges: pd.DataFrame) -> pd.DataFrame:
    test_nodes_list = set(test_edges[~test_edges['source'].isin(scores_df['node'])]['source'])
    test_nodes_list.update(test_edges[~test_edges['target'].isin(scores_df['node'])]['target'])

    labels = {}
    for n in test_nodes_list:
        neighbours = test_edges[test_edges['source'] == n]['target']
        neighbours.append(test_edges[test_edges['target'] == n]['source'])

        neighbours_df = scores_df[scores_df['node'].isin(neighbours)]
        n_score = neighbours_df.groupby('class')['score'].sum()

        duplicated_labels = n_score.duplicated(False)
        if True in duplicated_labels.values:
            n_label = 0
        else:
            n_label = n_score.idxmax()
        labels.update({n: n_label})

    labels_df = pd.DataFrame(labels.items(), columns=['node', 'label'])

    return labels_df


def main():
    edges = pd.read_csv(r'data/edges.csv')
    nodes = pd.read_csv(r'data/nodes.csv')
    node_dict = dict(zip(nodes['node'], nodes['class']))

    G = nx.from_pandas_edgelist(edges)
    nx.set_node_attributes(G, node_dict, 'label')

    scores_df = calculate_scores(G)

    test_edges = pd.read_csv(r'data/test_edges.csv')
    test_nodes = pd.read_csv(r'data/test_nodes.csv')

    test_labels = fit_nodes(scores_df, test_edges)

    test_labels = test_labels.merge(test_nodes, how='left', left_on='node', right_on='node')

    # nx.draw(subgraph)
    # plt.show()

    print('done')


if __name__ == '__main__':
    main()
