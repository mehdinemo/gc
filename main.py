import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def main():
    edges = pd.read_csv(r'data/edges.csv')
    nodes = pd.read_csv(r'data/nodes.csv')
    node_dict = dict(zip(nodes['node'], nodes['class']))

    G = nx.from_pandas_edgelist(edges)
    nx.set_node_attributes(G, node_dict, 'label')

    labels = nx.get_node_attributes(G, 'label')
    degrees = G.degree()

    classes = list(set(labels.values()))
    # create generator
    subgraph_dic = {}
    for i in classes:
        sub_nodes = (
            node
            for node, data
            in G.nodes(data=True)
            if data.get('label') == classes[i]
        )
        subgraph = G.subgraph(sub_nodes)
        subgraph_dic.update({i: subgraph})

    for k, v in subgraph_dic.items():
        sub_deg = v.degree()


    nx.draw(subgraph)
    plt.show()

    print('done')


if __name__ == '__main__':
    main()
