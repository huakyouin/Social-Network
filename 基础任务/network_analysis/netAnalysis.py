import matplotlib.pyplot as plt
import networkx as nx
import math
import argparse

parser = argparse.ArgumentParser(description="Social Network Final Project net analysis")
parser.add_argument("--type", type=str, default="Facebook",
                    help="graph type: Facebook, RR, ER, WS, BA")
args = parser.parse_args()

# load the karate club graph
if args.type == "Facebook":
    G = nx.read_edgelist('../data/facebook_combined.txt', delimiter=' ', nodetype=int)
elif args.type == "RR":
    G = nx.random_graphs.random_regular_graph(44, 4039)
elif args.type == "ER":
    G = nx.random_graphs.erdos_renyi_graph(4039, 0.011)
elif args.type == "WS":
    G = nx.random_graphs.watts_strogatz_graph(4039, 44, 0.01)
elif args.type == "BA":
    G = nx.random_graphs.barabasi_albert_graph(4039, 22)

# Clustering
print(G)
Clus = nx.average_clustering(G)
print("Clustering:", Clus)

if nx.is_connected(G):
    # Average distance
    ave_len = nx.average_shortest_path_length(G)
    print("Average distance:", ave_len)

    # Diameter
    Diameter = nx.diameter(G)
    print("Diameter:", Diameter)

# Degree
# 存储度数相应点数
number = []
# 存储度数
degree = []
for i in nx.degree_histogram(G):
    number.append(i)
for j in range(len(nx.degree_histogram(G))):
    degree.append(j)
# 去掉number=0,并取log
logxy = {}
for i in range(len(degree)):
    if (number[i] != 0):
        logxy[math.log(degree[i])] = math.log(number[i])

# 作图
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title(str(args.type))
plt.xlabel("log(degree)")
plt.ylabel("log(number)")
plt.scatter(logxy.keys(), logxy.values(), c="red", s=10)
# plt.show()
plt.savefig("./figure/" + str(args.type) + ".png")