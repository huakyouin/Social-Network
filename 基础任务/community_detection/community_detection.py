import community
import infomap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from torch import seed
import argparse

def louvain(G, pos):
    # compute the best partition
    # partition = community_louvain.best_partition(G)
    # print(partition.values())
    # print(partition.keys())
    partition = nx.algorithms.community.louvain_communities(G, seed=2001)
    # print(partition)

    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', len(partition))
    nodes = []
    comm = []
    # print(len(partition))
    for i in range(len(partition)):
        # print(i)
        nodes = nodes + list(partition[i])
        comm = comm + [i] * len(partition[i])
    nx.draw_networkx_nodes(G, pos, nodes, node_size=15,
                           cmap=cmap, node_color=comm)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    partition_dict = {k: v for k, v in zip(nodes, comm)}
    # print(max(partition_dict.values()))
    modularity = community.modularity(partition_dict, G)
    print([max(partition_dict.values()) + 1, modularity])

    return len(partition)

def random_walk(G, pos):
    infomapWrapper = infomap.Infomap("--two-level --silent")
    for e in G.edges():
        infomapWrapper.addLink(*e)
    infomapWrapper.run()
    tree = infomapWrapper

    partition = {}
    for node in tree.nodes:
        partition[node.node_id] = node.module_id

    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=15,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    modularity = community.modularity(partition, G)
    print([max(partition.values()) + 1, modularity])

    return tree.numTopModules()


def label_propagation(G, pos):
    # compute the best partition
    partition = nx.algorithms.community.label_propagation_communities(G)
    keys, values = [], []
    for i, item in enumerate(partition):
        keys = keys + list(item)
        values = values + [i] * len(item)
    partition_dict = {k:v for k, v in zip(keys, values)}
    cmap = cm.get_cmap('viridis', max(partition_dict.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition_dict.keys(), node_size=15,
                           cmap=cmap, node_color=list(partition_dict.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    modularity = community.modularity(partition_dict, G)
    print([max(partition_dict.values()) + 1, modularity])
    return max(partition_dict.values()) + 1

def asyn_lpa(G, pos):
    # compute the best partition
    partition = nx.algorithms.community.asyn_lpa_communities(G, seed=7018)
    keys, values = [], []
    for i, item in enumerate(partition):
        keys = keys + list(item)
        values = values + [i] * len(item)
    partition_dict = {k:v for k, v in zip(keys, values)}
    cmap = cm.get_cmap('viridis', max(partition_dict.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition_dict.keys(), node_size=15,
                           cmap=cmap, node_color=list(partition_dict.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    modularity = community.modularity(partition_dict, G)
    print([max(partition_dict.values()) + 1, modularity])
    return max(partition_dict.values()) + 1

def asyn_fluidc(G, pos):
    # compute the best partition
    partition = nx.algorithms.community.asyn_fluidc(G, k=16, seed=7018)
    keys, values = [], []
    for i, item in enumerate(partition):
        keys = keys + list(item)
        values = values + [i] * len(item)
    partition_dict = {k:v for k, v in zip(keys, values)}
    cmap = cm.get_cmap('viridis', max(partition_dict.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition_dict.keys(), node_size=15,
                           cmap=cmap, node_color=list(partition_dict.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    modularity = community.modularity(partition_dict, G)
    print([max(partition_dict.values()) + 1, modularity])
    return max(partition_dict.values()) + 1

def greedy(G, pos):
    # compute the best partition
    partition = nx.algorithms.community.greedy_modularity_communities(G)
    keys, values = [], []
    for i, item in enumerate(partition):
        keys = keys + list(item)
        values = values + [i] * len(item)
    partition_dict = {k:v for k, v in zip(keys, values)}
    cmap = cm.get_cmap('viridis', max(partition_dict.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition_dict.keys(), node_size=15,
                           cmap=cmap, node_color=list(partition_dict.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    modularity = community.modularity(partition_dict, G)
    print([max(partition_dict.values()) + 1, modularity])
    return max(partition_dict.values()) + 1

def naive_greedy(G, pos):
    # compute the best partition
    partition = nx.algorithms.community.naive_greedy_modularity_communities(G)
    keys, values = [], []
    for i, item in enumerate(partition):
        keys = keys + list(item)
        values = values + [i] * len(item)
    partition_dict = {k:v for k, v in zip(keys, values)}
    cmap = cm.get_cmap('viridis', max(partition_dict.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition_dict.keys(), node_size=15,
                           cmap=cmap, node_color=list(partition_dict.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    modularity = community.modularity(partition_dict, G)
    print([max(partition_dict.values()) + 1, modularity])
    return max(partition_dict.values()) + 1

# load the graph
# G = nx.karate_club_graph()
parser = argparse.ArgumentParser(description="Social Network Final Project Community Detection")
parser.add_argument("--src", type=str, default=None,
                    help="Graph source")
args = parser.parse_args()
G = nx.read_edgelist(args.src, nodetype=int)
pos = nx.spring_layout(G)

# print("greedy test")
# greedy(G, pos)

# print("naive greedy test")
# naive_greedy(G, pos)

# print("async_fluidc test")
# asyn_fluidc(G, pos)

# print("label_propagation test")
# label_propagation(G, pos)

# print("async_lpa_communites test")
# asyn_lpa(G, pos)

print("louvian test")
louvain(G, pos)

print("infomap test")
random_walk(G, pos)
