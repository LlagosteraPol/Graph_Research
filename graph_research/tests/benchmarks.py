import sympy

from graph_research.core.graphtbox import *

class Benchmarks(object):

    @staticmethod
    def relpoly_ordered_cycles_console_benchmark(g_list):
        """
        This benchmark test the relpoly_ordered_cycles method
        :param g_list: List of Networkx graphs to test
        """
        p = sympy.symbols("p")
        for graph in g_list:
            basic = GraphRel.relpoly_binary_basic(graph)
            improved = GraphRel.relpoly_ordered_cycles(graph)

            if basic != improved:
                raise ("Error in graph: \n", list(graph.edges), "\n",
                       "Correct polynomial: \n", basic, "\n(p=0.6) = ", basic.subs({p: 0.6}), "\n",
                       "Current polynomial: \n", improved, "\n(p=0.6) = ", improved.subs({p: 0.6}, "\n"))

        print("All correct")

    @staticmethod
    def relpoly_binary_improved_console_benchmark(g_list, filter_depth=0):
        """
        This benchmark test the relpoly_binary_improved method
        :param g_list: List of Networkx graphs to test
        :param filter_depth: number of subgraphs that will be analyzed to determine if they're ordered cycles
        """
        p = sympy.symbols("p")
        for graph in g_list:
            basic = GraphRel.relpoly_binary_basic(graph)
            improved = GraphRel.relpoly_binary_improved(graph, filter_depth)

            if basic != improved:
                raise ValueError("Error in graph: \n", list(graph.edges), "\n",
                                 "Correct polynomial: \n", basic, "\n(p=0.6) = ", basic.subs({p: 0.6}), "\n",
                                 "Current polynomial: \n", improved, "\n(p=0.6) = ", improved.subs({p: 0.6}, "\n"))

        print("All correct")

    @staticmethod
    def relpoly_binary_improved_file_benchmark(path):
        """
        Checks the relpoly binary improved method but unless the console version, this benchmark reads the graphs from
        a file and writes the times and possible errors into a file.
        :param path: File to test path
        """
        print("Graph\tTotal Nodes\tCycles\tCycle nodes\tBasic\tImproved",
              file=open(path + "benchmark_results.txt", "w"))
        for filename in glob.glob(os.path.join(path, '*.dat')):
            graph_name = filename[len(path):-4]
            r_graph = AdjMaBox.read_simple_mtx(path + graph_name)
            cycles = nx.cycle_basis(nx.Graph(r_graph))
            total_cycle_nodes = 0
            for cycle in cycles:
                total_cycle_nodes += len(cycle)

            time_start1 = time.process_time()
            pol1 = GraphRel.relpoly_binary_basic(r_graph)
            time_elapsed1 = (time.process_time() - time_start1)

            time_start2 = time.process_time()
            pol2 = GraphRel.relpoly_binary_improved(r_graph)
            time_elapsed2 = (time.process_time() - time_start2)

            if pol1 == pol2:
                print(graph_name, "\t", len(r_graph.nodes), "\t", len(cycles), "\t",
                      total_cycle_nodes, "\t", time_elapsed1, "\t", time_elapsed2,
                      file=open(path + "benchmark_results.txt", "a"))

            else:
                print(graph_name, "\tError", file=open(path + "benchmark_results.txt", "a"))

    @staticmethod
    def glued_benchmark(path, m):
        """
        Benchmark to get the computing time to calculate a 2xm grid using relpoly binary basic method, the improved,
        and the improved with a depth level
        :param path: File to write the benchamrk resluts
        :param m: Size of the grid (2xm)
        """
        print("Graph\t\t Basic\t Improved\t New", file=open(path + "benchmark_glued_results.txt", "w"))
        for i in range(14, m + 1):
            grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, i))
            # AdjMaBox.plot(grid)
            print("Grid 2x", i)

            time_start_basic = time.process_time()
            GraphRel.relpoly_binary_basic(grid)
            time_elapsed_basic = (time.process_time() - time_start_basic)
            print("Time 1: ", time.process_time(), " - ", time_start_basic, " = ", time_elapsed_basic)

            time_start_improved = time.process_time()
            GraphRel.relpoly_binary_improved(grid)
            time_elapsed_improved = (time.process_time() - time_start_improved)
            print("Time 2: ", time.process_time(), " - ", time_start_improved, " = ", time_elapsed_improved)

            time_start_new = time.process_time()
            GraphRel.relpoly_binary_improved(grid, 0)
            time_elapsed_new = (time.process_time() - time_start_new)
            print("Time 3: ", time.process_time(), " - ", time_start_new, " = ", time_elapsed_new)

            graph_name = "Grid 2*" + str(i)
            print(graph_name, "\t", time_elapsed_basic, "\t", time_elapsed_improved, "\t", time_elapsed_new,
                  file=open(path + "benchmark_glued_results.txt", "a"))