from graph_research.core.utilities import *
from graph_research.core.graphtbox import *
import networkx as nx
import matplotlib.pyplot as plt


print("----------------------------------------------Hypercube------------------------------------------------")

for n in range(2, 4+1):
    hg = nx.hypercube_graph(n)

    start1 = time.time()
    poly = GraphRel.relpoly_binary_basic(hg)
    end1 = time.time()
    print(Utilities.polynomial2binomial(poly))
    print("Basic - Hypercube ", n, ":", end1 - start1)

    cycles_edges, bal_pedges = GraphTools.get_cycles_edges(hg)

    start2 = time.time()
    poly = GraphRel.relpoly_treecyc(hg, cycles_edges)
    end2 = time.time()
    print(Utilities.polynomial2binomial(poly))
    print("Advanced - Hypercube ", n, ":", end2 - start2)

    try:
        print("SP efficientcy compared to basic: ", round((end1-start1)*100/(end2-start2), 3), "%")

    except (ZeroDivisionError):
        print("SP efficientcy compared to basic: ", round((end1 - start1) * 100, 3), "%")
print("-------------------------------------------------------------------------------------------------------")

print("----------------------------------------------Grid------------------------------------------------")

for n in range(2, 10+1):
    gg = nx.grid_graph(n, n)

    start1 = time.time()
    poly = GraphRel.relpoly_binary_basic(gg)
    end1 = time.time()
    print(Utilities.polynomial2binomial(poly))
    print("Basic - Grid ", n, ":", end1 - start1)

    cycles_edges, bal_pedges = GraphTools.get_cycles_edges(gg)

    start2 = time.time()
    poly = GraphRel.relpoly_treecyc(gg, cycles_edges)
    end2 = time.time()
    print(Utilities.polynomial2binomial(poly))
    print("Advanced - Grid ", n, ":",  end2 - start2)

    try:
        print("SP efficientcy compared to basic: ", round((end1-start1)*100/(end2-start2), 3), "%")

    except (ZeroDivisionError):
        print("SP efficientcy compared to basic: ", round((end1 - start1) * 100, 3), "%")
print("-------------------------------------------------------------------------------------------------------")


print("-------------------------------------------Circular Ladder---------------------------------------------")

for n in range(2, 10+1):
    gg = nx.circular_ladder_graph(n)

    start1 = time.time()
    poly = GraphRel.relpoly_binary_basic(gg)
    end1 = time.time()
    print(Utilities.polynomial2binomial(poly))
    print("Basic - Grid ", n, ":", end1 - start1)

    cycles_edges, bal_pedges = GraphTools.get_cycles_edges(gg)

    start2 = time.time()
    poly = GraphRel.relpoly_treecyc(gg, cycles_edges)
    end2 = time.time()
    print(Utilities.polynomial2binomial(poly))
    print("Advanced - Grid ", n, ":",  end2 - start2)

    try:
        print("SP efficientcy compared to basic: ", round((end1-start1)*100/(end2-start2), 3), "%")

    except (ZeroDivisionError):
        print("SP efficientcy compared to basic: ", round((end1 - start1) * 100, 3), "%")
print("-------------------------------------------------------------------------------------------------------")


print("----------------------------------------------Complete------------------------------------------------")

for n in range(2, 10+1):
    gg = nx.grid_graph(n, n)

    start1 = time.time()
    poly = GraphRel.relpoly_binary_basic(gg)
    end1 = time.time()
    print(Utilities.polynomial2binomial(poly))
    print("Basic - Complete ", n, ":", end1 - start1)

    cycles_edges, bal_pedges = GraphTools.get_cycles_edges(gg)

    start2 = time.time()
    poly = GraphRel.relpoly_treecyc(gg, cycles_edges)
    end2 = time.time()
    print(Utilities.polynomial2binomial(poly))
    print("Advanced - Complete ", n, ":",  end2 - start2)

    try:
        print("SP efficientcy compared to basic: ", round((end1-start1)*100/(end2-start2), 3), "%")

    except (ZeroDivisionError):
        print("SP efficientcy compared to basic: ", round((end1 - start1) * 100, 3), "%")
print("-------------------------------------------------------------------------------------------------------")