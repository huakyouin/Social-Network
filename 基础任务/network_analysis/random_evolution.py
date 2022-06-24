import networkx as nx
import matplotlib.pyplot as plt
import math

n = 1000
G = nx.random_graphs.erdos_renyi_graph(n, 0)
pos = nx.spring_layout(G)

for c in [0.01, 0.1, 1, 10]:
    G = nx.random_graphs.erdos_renyi_graph(n, c/(n-1))

    subG = G.subgraph(max(nx.connected_components(G),key=len))
    
    print("==========================")
    # Average degree
    ave_deg = len(G.edges())*2/n
    print("Average degree:", ave_deg)

    Clus = nx.average_clustering(subG)
    print("Clustering:", Clus)

    # Average distance
    ave_len = nx.average_shortest_path_length(subG)
    print("Average distance:", ave_len)

    # Diameter
    Diameter = nx.diameter(subG)
    print("Diameter:", Diameter)

    # Size
    size = len(subG.nodes())
    print("Size:", size)


    nx.draw_networkx_nodes(G, pos, node_size=15)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()