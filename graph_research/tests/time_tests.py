from graph_research.core.utilities import *
from graph_research.core.graphtbox import *
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum

class GType(Enum):
    Hypercube = 1
    Grid = 2
    CircularLadder = 3
    Complete = 4

def compare_times(gtype, n_max):
    print("----------------------------------------------", gtype.name, "------------------------------------------------")
    times = dict()
    for n in range(2, n_max + 1):
        print("\n-----------> n =", n)
        if gtype == GType.Hypercube:
            g = nx.hypercube_graph(n)
        elif gtype == GType.Grid:
            g = nx.grid_graph([n,n])
        elif gtype == GType.CircularLadder:
            g = nx.circular_ladder_graph(n)
        elif gtype == GType.Complete:
            g = nx.complete_graph(n)

        start1 = time.time()
        poly = GraphRel.relpoly_binary_basic(g)
        end1 = time.time()
        t1 = end1 - start1
        print(Utilities.polynomial2binomial(poly))
        print("Basic - ", gtype.name, " ", n, ":", t1)

        start2 = time.time()
        poly = GraphRel.relpoly_binary_improved(g)
        end2 = time.time()
        t2 = end2 - start2
        print(Utilities.polynomial2binomial(poly))
        print("Advanced - ", gtype.name," ", n, ":", t2)

        times[n] = (t1, t2)
        try:
            print("SP efficientcy compared to basic: ", round(t1 * 100 / t2, 3), "%")

        except (ZeroDivisionError):
            print("SP efficientcy compared to basic: ", round(t1 * 100, 3), "%")
    print("-------------------------------------------------------------------------------------------------------")
    return times

def ctime_hypercube(n_max):
    compare_times(GType.Hypercube, n_max)

def ctime_grid(n_max):
    compare_times(GType.Grid, n_max)

def ctime_circularladder(n_max):
    compare_times(GType.CircularLadder, n_max)

def ctime_complete(n_max):
    compare_times(GType.Complete, n_max)

#ctime_hypercube(3)
#ctime_grid(3)
#ctime_circularladder(3)
ctime_complete(3)
