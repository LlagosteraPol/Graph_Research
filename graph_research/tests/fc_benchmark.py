import collections

from graph_research.core.graphtbox import *

coeffs_old = collections.OrderedDict()
coeffs_new = collections.OrderedDict()

os.chdir("..")
path = os.getcwd() + "/data"

min_n_nodes = 8
max_n_nodes = 8

for n in range(min_n_nodes, max_n_nodes + 1):
    g_list = nx.read_graph6(path + "/graph6/" + str(n) + "n_FairCake.g6")

    for g in g_list:
        poly = GraphRel.relpoly_binary_improved(g, 0)
        bin_poly, bin_coefficients = Utilities.polynomial2binomial(poly)
        coeffs_old[(n, g.number_of_edges())] = list(map(int, bin_coefficients))


#CakeRel.cake_rel()

print(coeffs_old)
