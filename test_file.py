import networkx as nx
import pandas as pd
import community
import json
from text_tools import TextTools
from prepare_data import PrepareData
from clustering_manipulator import ClusteringManipulator


def main():
    tt = TextTools()
    pr = PrepareData()
    cm = ClusteringManipulator()
    #
    # delete_similar_data = False
    # weight = 'jaccard_sim'
    #
    # with open(r'B:\Graph Classification\data\json\vohoush.json', encoding='utf-8') as f:
    #     data = json.load(f)
    #
    # messages_df = pd.DataFrame(data['messages'])
    #
    # messages_df = messages_df.sample(frac=0.1)
    #
    # messages_df['id'] = messages_df['id'].astype('int64').astype(str)
    #
    # messages_df['clean_text'] = tt.normalize_texts(messages_df['text'])
    #
    # allkeywords = tt._text_to_allkeywords(messages_df)
    # graph = tt._create_graph(allkeywords)
    #
    # # graph.to_csv('data/graph_vohoush.csv', index=False)
    #
    # graph['source'] = graph['source'].astype('int64')
    # graph['target'] = graph['target'].astype('int64')
    #
    # data_sim = pr._jaccard_sim(graph)
    # if delete_similar_data:
    #     print('delete similar data...')
    #     data_sim = pr._sim_nodes_detector(data_sim)
    #
    # # data_sim['source'] = data_sim['source'].astype(str)
    # # data_sim['target'] = data_sim['target'].astype(str)
    #
    # print('creating graph...')
    # G = nx.from_pandas_edgelist(data_sim, source='source', target='target', edge_attr=True)
    # G.remove_edges_from(nx.selfloop_edges(G))
    # print(f'graph created with {len(G)} nodes and {G.number_of_edges()} edges.')
    #
    # # prune graph with max weight
    # sim = pr._adj_matrix(G, weight)
    # print('prune graph...')
    # G_p = pr._prune_max(sim)
    # sim_p = pr._adj_matrix(G)
    # print('graph pruned!')
    #
    # sim_p = pr._adj_matrix(G_p)
    # sim_p.to_csv('data/sim_p_vohoush.csv')
    # sim.to_csv('data/sim_vohoush.csv')

    sim = pd.read_csv('data/sim_vohoush.csv', index_col=0)
    sim_p = pd.read_csv('data/sim_p_vohoush.csv', index_col=0)

    G = nx.from_numpy_matrix(sim.values)
    G_p = nx.from_numpy_matrix(sim_p.values)

    sim_nodes = dict(zip(G.nodes, sim.index))
    sim_p_nodes = dict(zip(G_p.nodes, sim_p.index))

    sim_nodes = pd.DataFrame.from_dict(sim_nodes, orient='index')
    sim_p_nodes = pd.DataFrame.from_dict(sim_p_nodes, orient='index')
    sim_nodes.to_csv('data/sim_nodes.csv')
    sim_p_nodes.to_csv('data/sim_p_nodes.csv')
    # sim_nodes.reset_index(inplace=True)
    # sim_p_nodes.reset_index(inplace=True)
    # nx.relabel_nodes(G, sim_nodes, copy=False)
    # nx.relabel_nodes(G_p, sim_p_nodes, copy=False)

    partitions = community.best_partition(G)
    partitions_p = community.best_partition(G_p)
    partitions_fun = cm.clustering_matrix(G, sim, True)
    partitions_fun_p = cm.clustering_matrix(G_p, sim_p, True)

    res = pd.DataFrame.from_dict(partitions, orient='index')
    res = res.merge(pd.DataFrame.from_dict(partitions_p, orient='index'), how='inner', left_index=True,
                    right_index=True)
    res = res.merge(partitions_fun, how='inner', left_index=True, right_index=True)
    res = res.merge(partitions_fun_p, how='inner', left_index=True, right_index=True)

    res.columns = ['community', 'community_p', 'community_f', 'community_f_p']
    # res.reset_index(inplace=True)
    res.to_csv('data/vohoush_res.csv', index=True)

    print('done')


if __name__ == '__main__':
    main()
