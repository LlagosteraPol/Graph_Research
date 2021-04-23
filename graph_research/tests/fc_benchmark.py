import collections

from graph_research.core.graphtbox import *

coeffs_old = collections.OrderedDict()
coeffs_new = collections.OrderedDict()

os.chdir("..")
path = os.getcwd() + "/data"

min_n_nodes = 8
max_n_nodes = 21

for n in range(min_n_nodes, max_n_nodes + 1):
    g_list = nx.read_graph6(path + "/graph6/" + str(n) + "n_FairCake.g6")

    for g in g_list:
        poly = GraphRel.relpoly_binary_improved(g, 0)
        bin_poly, bin_coefficients = Utilities.polynomial2binomial(poly)
        coeffs_old[(n, g.number_of_edges())] = list(map(int, bin_coefficients))
        test = CakeRel.get_fc_cpaths(g)
        coeffs_new[(n, g.number_of_edges())] = CakeRel.cake_rel(CakeRel.get_fc_cpaths(g))


for key in coeffs_old.keys():
    if coeffs_old[key] == coeffs_new[key]:
        print(str(key) + " correct")
    else:
        print(str(key) + " NOT correct")


print(coeffs_old)
print(coeffs_new)
