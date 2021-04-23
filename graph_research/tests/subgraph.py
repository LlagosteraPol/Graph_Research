import networkx as nx
import matplotlib.pyplot as plt

"""
Test file: testing subgraphs
"""

g1 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)])
g1_2 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4), (0, 4)])
g2 = nx.Graph([(5, 6), (6, 7), (7, 5)])
g3 = nx.Graph([(5, 6), (6, 7), (7, 8), (8, 5)])

g4 = nx.Graph([(0, 3), (0, 4), (0, 5), (1, 3), (1, 5), (2, 4), (2, 5)])
g5 = nx.Graph([(0, 3), (0, 4), (1, 3), (1, 5), (2, 4), (2, 5)])

g6 = nx.Graph([(0, 3), (0, 4), (0, 5), (1, 4), (1, 5), (1, 6), (2, 5), (2, 6), (2, 7), (3, 6), (3, 7), (4, 7)])
g7 = nx.Graph([(0, 4), (0, 6), (0, 7), (1, 4), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7), (4, 5)])

g8 = nx.Graph([(0, 3), (0, 4), (0, 5), (1, 3), (1, 5), (2, 4), (2, 5)])

sub = g6.subgraph(g7.edges)
print(sub.edges)

nx.draw(g4)
plt.show()

GM = nx.algorithms.isomorphism.GraphMatcher(g1_2, g2)
print(GM.subgraph_is_isomorphic())



for subgraph in GM.subgraph_isomorphisms_iter():
    print (subgraph)

