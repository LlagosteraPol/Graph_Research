from Main.graphtbox import *
import time
import os

class TestMethod(Enum):
    RP_Basic = 1  # Reliability Polynomial Basic
    RP_Improved = 2  # Reliability Polynomial Improved
    RP_2xl = 3  # # Reliability Polynomial Grid 2xl


def time_test_relpoly_binary_basic(g):
    GraphRel.relpoly_binary_basic(g)


def time_test_relpoly_binary_improved(g):
    GraphRel.relpoly_binary_improved(g, False)


def time_test_relpoly_binary_new(g):
    GraphRel.relpoly_binary_improved(g, True)


def time_test_optimal_glued_cycles(g):
    GraphRel.relpoly_grid_2xl(g)


def time_test_relpoly_ordered_cycles(g):
    GraphRel.relpoly_ordered_cycles(g)


def time_test_p2xl_reduction(g):
    GraphRel.relpoly_p2xl(g)


def time_tests(g):
    time_start = time.process_time()
    time_test_relpoly_binary_basic(g)
    time_elapsed = (time.process_time() - time_start)
    print("\nRelpoly binary basic - Time: ", time_elapsed)

    time_start = time.process_time()
    time_test_relpoly_ordered_cycles(g)
    time_elapsed = (time.process_time() - time_start)
    print("\nRelpoly ordered cycles - Time: ", time_elapsed)

    time_start = time.process_time()
    time_test_p2xl_reduction(g)
    time_elapsed = (time.process_time() - time_start)
    print("\nP2xl Reduction - Time: ", time_elapsed)

    time_start = time.process_time()
    time_test_optimal_glued_cycles(g)
    time_elapsed = (time.process_time() - time_start)
    print("\nOptimal glued cycles - Time: ", time_elapsed)


def test_relpoly_binary_basic(g, prob=0.6, result=False, timer=False):
    print("Graph: \n", list(g.edges), "\n")
    p = sympy.symbols("p")

    time_start = time.process_time()
    pol = GraphRel.relpoly_binary_basic(g)
    time_elapsed = (time.process_time() - time_start)

    if result:
        print("\nRelpoly binary basic: \n",
              pol, " -> (p = ",prob,") = ", pol.subs({p: prob}))

    if timer:
        print("Time: ", time_elapsed)

    return pol


def test_relpoly_binary_improved(g, filter_depth = 0, prob=0.6, result=False, timer=False):
    print("Graph: \n", list(g.edges), "\n")
    p = sympy.symbols("p")

    time_start = time.process_time()
    pol = GraphRel.relpoly_binary_improved(g, filter_depth)
    time_elapsed = (time.process_time() - time_start)

    if result:
        print("Relpoly binary improved:",
        pol, " -> (p = ",prob,") = ", pol.subs({p: prob}), "\n")

    if timer:
        print("Time: ", time_elapsed)

    return pol

#def bench_relpoly_binary_improved(n_min, n_max):


def test_relpoly_multitree(g=None, prob=0.6, timer=False):
    p = sympy.symbols("p")

    time_start = time.process_time()
    pol = GraphRel.relpoly_multitree(g)
    time_elapsed = (time.process_time() - time_start)

    print("\nRelpoly multitree:     \n", pol, "\n= ", pol.subs({p: prob}))
    if timer:
        print("Time: ", time_elapsed)

    return pol


def test_relpoly_multicycle(g=None, prob=0.6, timer=False):
    p = sympy.symbols("p")

    time_start = time.process_time()
    pol = GraphRel.relpoly_multicycle(g)
    time_elapsed = (time.process_time() - time_start)

    print("\nRelpoly multicycle:     \n", pol, "\n= ", pol.subs({p: prob}))
    if timer:
        print("Time: ", time_elapsed)

    return pol


def test_relpoly_multi_treecyc(g=None, prob=0.6, timer=False):
    p = sympy.symbols("p")

    time_start = time.process_time()
    pol = GraphRel.relpoly_multi_treecyc(g)
    time_elapsed = (time.process_time() - time_start)

    print("\nRelpoly multitreecyc:     \n", pol, "\n= ", pol.subs({p: prob}))
    if timer:
        print("Time: ", time_elapsed)

    return pol


def test_relpoly_ordered_cycles(g=None, prob=0.6, timer=False):
    p = sympy.symbols("p")

    time_start = time.process_time()
    pol = GraphRel.relpoly_ordered_cycles(g)
    time_elapsed = (time.process_time() - time_start)

    print("\nRelpoly Ordered Cycles:     \n", pol, "\n= ", pol.subs({p: prob}))
    if timer:
        print("Time: ", time_elapsed)

    return pol


def test_optimal_glued_cycles(g=None, prob=0.6, timer=False):
    p = sympy.symbols("p")

    time_start = time.process_time()
    pol = GraphRel.relpoly_grid_2xl(g)
    time_elapsed = (time.process_time() - time_start)

    print("\nOptimal glued cycles:\n", pol, "\n= ", pol.subs({p: prob}))
    if timer:
        print("Time: ", time_elapsed)

    return pol


def test_relPoly_glued_cycles(g=None, prob=0.6, timer=False):
    p = sympy.symbols("p")

    time_start = time.process_time()
    pol = GraphRel.relPoly_glued_cycles(g)
    time_elapsed = (time.process_time() - time_start)

    print("\nRelpoly glued cycles:\n", pol, "\n= ", pol.subs({p: prob}))
    if timer:
        print("Time: ", time_elapsed)

    return pol


def run(focus=False):
    p = sympy.symbols("p")
    g_prob = 0.6
    show_time = True
    bench_path = os.getcwd() + "/Data/Benchmarks/"
    g6_path = os.getcwd() + "/Data/Graph6/"

    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(2,4))
    GraphTools.plot(graph)

    # TestMethod.RP_Basic

    if focus:
        Benchmarks.relpoly_binary_improved_console_benchmark([graph])

    else:
        #test_relPoly_glued_cycles(None, g_prob, timer=False)
        Benchmarks.relpoly_binary_improved_console_benchmark([graph])


# ----MAIN----
run(True)


