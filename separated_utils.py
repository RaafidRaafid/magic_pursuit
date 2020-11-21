import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

finalize = True
cases = 10
nodes = 15
pd = 3
pr = 2

deg = []
for i in range(nodes):
    deg.append(np.random.randint(2, pd+1))
if sum(deg) % 2 == 1:
    maxx = max(deg)
    max_index = deg.index(maxx)
    deg[max_index] = deg[max_index] - 1
G = nx.random_degree_sequence_graph(deg, seed=42)
adj = np.zeros([nodes, nodes])
for edge in G.edges():
    adj[edge[0]][edge[1]] = 1.0
    adj[edge[1]][edge[0]] = 1.0
for i in range(nodes):
    adj[i][i] = 1.0

if finalize:
    file = open("data/adj.txt", "w+")
    for i in range(nodes):
        for j in range(nodes):
            file.write(str(adj[i][j]))
            if j < nodes - 1:
                file.write(" ")
        if i < nodes - 1:
            file.write("\n")

    file.close()

for k in range(cases):
    available = list(range(nodes))
    pd_idx = []
    pr_idx = []

    for i in range(pd):
        idx = np.random.randint(0, nodes - i)
        pd_idx.append(available[idx])
        available.pop(idx)
    for i in range(pr):
        idx = np.random.randint(0, nodes - pd - i)
        pr_idx.append(available[idx])
        available.pop(idx)

    color_map = ["yellow"] * nodes
    for idx in pd_idx:
        color_map[idx] = "red"
    for idx in pr_idx:
        color_map[idx] = "green"

    if finalize:
        file = open("data/" + str(k) + "pd.txt", "w+")
        for i in range(pd):
            file.write(str(pd_idx[i]))
            if i < pd - 1:
                file.write(" ")
        file.close()
        file = open("data/" + str(k) + "pr.txt", "w+")
        for i in range(pr):
            file.write(str(pr_idx[i]))
            if i < pd - 1:
                file.write(" ")

    nx.draw_networkx(G, node_color=color_map)
    plt.show()
