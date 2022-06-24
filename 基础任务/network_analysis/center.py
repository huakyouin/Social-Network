from community import community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import math
import numpy as np

# degree_centrality
def degree(G, pos):
    degree_dict = nx.degree_centrality(G)
    max_center_value = max(degree_dict.values())
    color = []
    for i in degree_dict.keys():
        if degree_dict[i] > max_center_value*0.5:
            color.append("r")
        elif degree_dict[i] > max_center_value*0.25:
            color.append("c")
        else:
            color.append([0.5, 0.5, 0.5])  # grey
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("degree centrality")
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.show()
    return True

# eigenvector_centrality
def eigen(G, pos):
    dict = nx.eigenvector_centrality(G)
    max_center_value = max(dict.values())
    color = []
    for i in dict.keys():
        if dict[i] > max_center_value*0.5:
            color.append("r")
        elif dict[i] > max_center_value*0.25:
            color.append("c")
        else:
            color.append([0.5, 0.5, 0.5])  # grey
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("eigenvector centrality")
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    return True

# Katz_centrality
def katz(G, alpha, pos, beta = 1):
    dict = nx.katz_centrality(G, alpha, beta)
    max_center_value = max(dict.values())
    color = []
    for i in dict.keys():
        if dict[i] > max_center_value*0.5:
            color.append("r")
        elif dict[i] > max_center_value*0.25:
            color.append("c")
        else:
            color.append([0.5, 0.5, 0.5])  # grey
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("Katz centrality")
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    return True

# pagerank_centrality
def pagerank(G, pos, alpha = 0.85):
    dict = nx.pagerank(G, alpha)
    max_center_value = max(dict.values())
    color = []
    for i in dict.keys():
        if dict[i] > max_center_value*0.5:
            color.append("r")
        elif dict[i] > max_center_value*0.25:
            color.append("c")
        else:
            color.append([0.5, 0.5, 0.5])  # grey
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("pagerank centrality")
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    return True

# betweenness_centrality
def between(G, pos):
    dict = nx.betweenness_centrality(G)
    max_center_value = max(dict.values())
    color = []
    for i in dict.keys():
        if dict[i] > max_center_value*0.5:
            color.append("r")
        elif dict[i] > max_center_value*0.25:
            color.append("c")
        else:
            color.append([0.5, 0.5, 0.5])  # grey
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("betweenness centrality")
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    return True

# closeness_centrality
def close(G, pos):
    dict = nx.closeness_centrality(G)
    max_center_value = max(dict.values())
    color = []
    for i in dict.keys():
        if dict[i] > max_center_value*0.5:
            color.append("r")
        elif dict[i] > max_center_value*0.25:
            color.append("c")
        else:
            color.append([0.5, 0.5, 0.5])  # grey
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("closeness centrality")
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color=color)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    return True


# load the karate club graph
G = nx.read_edgelist('../data/facebook_combined.txt', nodetype=int)
# G = nx.karate_club_graph()
partition = community_louvain.best_partition(G)
subnode = []
for i in partition.keys():
    if partition[i] == 1:
        subnode.append(i)
G = G.subgraph(subnode)

pos = nx.spring_layout(G)

degree(G, pos)

eigen(G, pos)

w, v = np.linalg.eig(nx.to_numpy_matrix(G))
rho = max(abs(w))
alpha = 0.85/rho
katz(G, alpha, pos)

pagerank(G, pos)

between(G, pos)

close(G, pos)
