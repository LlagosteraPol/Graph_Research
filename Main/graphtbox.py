import os
import sys
import numpy as np
import hashlib
import networkx as nx
from networkx.algorithms import isomorphism
import sympy
from sympy import roots
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
import itertools as itt
import copy
from math import modf
from math import factorial
from collections import OrderedDict
from collections import Counter
from random import *
from enum import Enum
from collections import defaultdict
import glob
import time
import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import types

Base = declarative_base()
class Table_Graph(Base):
    __tablename__ = 'Graphs'
    g6_hash = db.Column(db.types.String(), primary_key=True, index=True)
    g6 = db.Column(db.types.String(), unique=True, nullable=False)
    #g6_hash = db.Column(db.types.String())
    #g6_id = db.Column(db.types.String(), primary_key=True, index=True)
    nodes = db.Column(db.types.INTEGER)
    edges = db.Column(db.types.INTEGER)
    hamiltonian = db.Column(db.types.BOOLEAN)
    hamiltonian_cycle = db.Column(db.types.String)
    graph_edges = db.Column(db.types.String)
    avg_polynomial = db.Column(db.types.FLOAT)
    polynomial = db.Column(db.types.String)
    spanning_trees = db.Column(db.INTEGER)
    edge_connectivity = db.Column(db.INTEGER)
    min_k2_edge_cuts = db.Column(db.INTEGER)
    automorphisms = db.Column(db.INTEGER)
    diameter = db.Column(db.INTEGER)
    probability_01 = db.Column(db.types.FLOAT)
    probability_02 = db.Column(db.types.FLOAT)
    probability_03 = db.Column(db.types.FLOAT)
    probability_04 = db.Column(db.types.FLOAT)
    probability_05 = db.Column(db.types.FLOAT)
    probability_06 = db.Column(db.types.FLOAT)
    probability_07 = db.Column(db.types.FLOAT)
    probability_08 = db.Column(db.types.FLOAT)
    probability_09 = db.Column(db.types.FLOAT)


class GraphType(Enum):
    Tree = 1
    PureCycle = 2
    OrderedCycles = 3
    Others = 4


class FormatType(Enum):
    CSV = 1
    JSON = 2
    HTML = 3
    Excel = 4
    SQL = 5


class Utilities(object):

    @staticmethod
    def input_g6_file(message):
        """
        Simple function to ask the user to input a .g6 file name.
        :param message: Message that will be shown to the user.
        :return: List of the read networkx graphs and the given file name
        """
        path = os.getcwd() + "/Data/Graph6/"

        while True:
            file_name = input(message + "\n")
            if os.path.isfile(path + file_name + ".g6"):
                g_list = nx.read_graph6(path + file_name + ".g6")
                # If the read graphs is only one, wrap it with a list
                if type(g_list) is not list:
                    g_list = [g_list]
                return g_list, file_name

            else:
                print("The file ", file_name, " doesn't exist, please make sure that is in the folder ", path)

    @staticmethod
    def input_number(message):
        """
        Simple function to ask the user to input a number.
        :param message: Message that will be shown to the user.
        :return: Users given number <float>
        """
        while True:
            try:
                user_input: float = float(input(message))
            except ValueError:
                print("The input is not an number, try again.")
                continue
            else:
                return user_input

    @staticmethod
    def input_data_format(message):
        """
        Simple function to ask the user to input the data format of the desired file.
        :param message: Message that will be shown to the user.
        :return: enum<FormatType>
        """
        while True:
            format_type = int(input(message + "\n"))
            if format_type == 1:
                return FormatType.CSV

            elif format_type == 2:
                return FormatType.Excel

            elif format_type == 3:
                return FormatType.HTML

            elif format_type == 4:
                return FormatType.JSON

            elif format_type == 5:
                return FormatType.SQL

            else:
                print("Not a valid number")

    @staticmethod
    def ask_yes_no(message):
        """
        Simple function to ask the user to input y/n.
        :param message: Message that will be shown to the user.
        :return: boolean (y-True, n-False)
        """
        query = input("\n" + message + " Y/N\n")

        answer = query[0].lower()

        while True:
            if query == '' or answer not in ['y', 'n']:
                print('Please answer y/n')
            else:
                break

        return True if answer == 'y' else False

    @staticmethod
    def number_combination(n, k):
        """
        Determines how many combinations can be done.
        :param n: Number of elements.
        :param k: Number of elements to combine.
        :return: Number of combinations with the given specifications.
        """
        if k < 0:
            return 0

        return factorial(n) / factorial(k) / factorial(n - k)

    @staticmethod
    def remove_dup_with_order(text):
        """
        Remove the duplicate letters of the given string keeping the same order of characters.
        :param text: String to remove the duplicate characters.
        :return: String without duplicate characters.
        """
        return "".join(OrderedDict.fromkeys(text))

    @staticmethod
    def remove_dup_without_order(text):
        """
        Remove the duplicate characters of the given string without any order.
        :param text: String to remove the duplicate characters.
        :return: String without duplicate characters.
        """
        return "".join(set(text))

    @staticmethod
    def intersection(lst1, lst2):
        """
        Gives the intersection between the two given lists. Complexity O(n)
        :param lst1: First list
        :param lst2: Second list
        :return: A list containing the intersection of the given lists
        """
        # Use of hybrid method
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    @staticmethod
    def difference(lst1, lst2):
        """
        Gives the difference between the two given lists
        :param lst1: First list
        :param lst2: Second list
        :return: A list containing the diffenrence of the given lists
        """
        return list(set(lst1) - set(lst2))

    @staticmethod
    def set_combinations(sets, n, incl_excl_intersect=False):
        """
        Combination of the elements of the given sets.
        :param sets: Sets to combine.
        :param n: Num of elements to combine xCn
        :param incl_excl_intersect: Special flag for inclusion_exclusion algorithm
        :return: List of all combinations.
        """
        raw_combinations = []

        for subset in itt.combinations(sets, n):
            raw_combinations.append(subset)

        combinations = []

        for raw_comb in raw_combinations:
            if incl_excl_intersect:
                raw_comb = list(itt.chain.from_iterable(raw_comb))  # Join lists
            raw_comb = set().union(raw_comb)  # Remove duplicate elements
            combinations.append(raw_comb)

        return combinations

    @staticmethod
    def subtract_dict_values(dict1, dict2):
        """
        Subtraction of the two given dictionaries.
        :param dict1: First dictionary to subtract.
        :param dict2: Second dictionary to subtract.
        :return: Dictionary product of the subtraction of the given dictionaries.
        """
        for key in dict1:
            if key in dict2:
                dict1[key] = dict1[key] - dict2[key]
        return dict1

    @staticmethod
    def dict_union(dict1, dict2):
        """
        Union of the two given dictionaries.
        :param dict1: First dictionary to union.
        :param dict2: Second dictionary to union.
        :return: Dictionary product of the union of the given dictionaries.
        """
        xx = Counter(dict1)
        yy = Counter(dict2)
        xx.update(yy)

        return xx

    @staticmethod
    def flatten_nested_list(lst):
        """
        Flatten the given list.
        :param lst: List to be flattened.
        :return: Flattened list.
        """
        return [y for x in lst for y in x]

    @staticmethod
    def list_to_dict(lst, value):
        """
        Convert the given list to a dictionary where all the keys have the given value.
        :param lst: List to be converted to dictionary (the values of the list will
        be the keys).
        :param value: The value of all the keys.
        :return: Dictionary where keys-> lst and value-> value.
        """
        return {k: value for k in lst}

    @staticmethod
    def compare_coefficients(coefficients1, coefficients2):
        """
        Compare each coefficient of the first polynomial with each coefficient of the second one.
        If all the coefficients of the first polynomial are greater than the ones of the second polynomial
        then return True, else return False
        :param coefficients1: First polynomial coefficients
        :param coefficients2: Second polynomial coefficients
        :return: boolean
        """
        if len(coefficients1) != len(coefficients2):
            raise Exception("Both polynomials must have the same degree.")

        for i in range(0, len(coefficients1)):

            if abs(coefficients1[i]) < abs(coefficients2[i]):
                if coefficients1[i] == coefficients2[i] and coefficients1 != 0:
                    return False

        return True

    @staticmethod
    def polynomial2binomial(polynomial):
        """
        This method transforms the given polynomial to its binomial form.
        :param polynomial: Polynomial to convert to binomial form
        :return: Binomial Polynomial, coefficients
        """
        p = sympy.symbols('p')

        # Assuring that the given polynomial is in the right class
        if type(polynomial) is not sympy.polys.polytools.Poly:
            polynomial = sympy.poly(polynomial)

        binomial = 0
        degree = sympy.degree(polynomial, p)

        # Get coefficients (and round the ones really close to 0)
        coefficients = Utilities.refine_polynomial_coefficients(polynomial)
        coefficients = np.trim_zeros(coefficients)  # Delete all right zeroes
        n_coeff = len(coefficients)

        # Get binomial coefficients
        aux = n_coeff
        aux_degree = degree
        for i in range(1, n_coeff):
            coeff2expand = coefficients[-i]
            expanded = sympy.Poly(coeff2expand * p ** (aux_degree - n_coeff + 1) * (1 - p) ** (aux - 1))
            tmp_coefficients = expanded.all_coeffs()
            tmp_coefficients = np.trim_zeros(tmp_coefficients)
            tmp_n_coeff = len(tmp_coefficients)

            for z in range(2, tmp_n_coeff + 1):
                coefficients[(-z - (n_coeff - tmp_n_coeff))] -= tmp_coefficients[-z]

            aux -= 1
            aux_degree += 1

        # Assemble binomial polynomial
        aux_degree = degree
        for coeff in coefficients:
            binomial += coeff * p ** aux_degree * (1 - p) ** (degree - aux_degree)
            aux_degree -= 1

        return binomial, coefficients

    @staticmethod
    def refine_polynomial_coefficients(polynomial):
        """
        This method will round the coefficients of the polynomial that are almost zero (ex. at the order of e-10).
        When calculating Rel(G,p), if some of the coefficients are like this maybe is due to noise when calculating
        large amounts reliabilities with big polynomials.
        :param polynomial: polynomial to refine
        :return: refined polynomial
        """
        coefficients = polynomial.all_coeffs()

        refined_coefficients = list()
        for coefficient in coefficients:
            refined_coefficients.append(round(coefficient, 0))

        return refined_coefficients


class AdjMaBox(object):

    @staticmethod
    def read_simple_mtx(path):
        """
        Read an adjacency matrix from a .dat file.
        :param path: <string> Folder containing the .dat file.
        :return: networkx graph.
        """
        matrix = np.loadtxt(path + '.dat')  # Read from file
        np_mtx = np.matrix(matrix, dtype='int64')  # Convert to a matrix
        # Convert to a networkx Graph
        return nx.from_numpy_matrix(np_mtx, parallel_edges=True, create_using=nx.MultiGraph())

    @staticmethod
    def write_matrix(g, path):
        """
        Transform the given networkx graph into an adjacency matrix and write it in a .dat file.
        :param g: Networkx graph.
        :param path: folder where the input file will be written.
        """
        matrix = nx.to_numpy_matrix(g)
        np.savetxt(path + '.dat', matrix, fmt="%d")  # To decimal format

    @staticmethod
    def ask_to_read():
        """
        Ask for reading a file that contains an adjacency matrix.
        :return: Networkx graph.
        """
        path = input("\nInput the adjacency matrix file name\n")
        return AdjMaBox.read_simple_mtx(path)


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


class GraphTools(object):

    @staticmethod
    def plot(graph):
        """
        Simple method to plot the given graph.
        :param graph: networkx graph.
        """
        nx.draw(graph)
        plt.show()

    @staticmethod
    def check_one_edge_less_subgraph_isomorphism(super_graph, sub_graph):
        """
        This method checks if the hamiltonian given graph without one chord is isomorphic with the second given graph.
        :param super_graph: Bigger graph
        :param sub_graph: Smaller graph
        :return: True if it's hamiltonian or otherwise, False
        """

        for edge in super_graph.edges:
            tmp_graph = copy.deepcopy(super_graph)
            tmp_graph.remove_edge(edge[0], edge[-1])
            gm = nx.algorithms.isomorphism.GraphMatcher(tmp_graph, sub_graph)
            if gm.is_isomorphic():
                return True

        return False

    @staticmethod
    def check_isomorphism(g1, g2):
        """
        Simple method to check isomorphism between two graphs
        :param g1: Networkx Graph
        :param g2: Networkx Graph
        :return: True if they're isomorphic, false otherwise
        """
        gm = nx.algorithms.isomorphism.GraphMatcher(g1, g2)
        if gm.is_isomorphic():
            return True
        else:
            return False

    @staticmethod
    def minimum_k_edges_cutsets(g, k):
        """
        Method to calculate the minimum cut-sets in a graph with k edges.
        :param g: Networkx Graph
        :param k: K edge cut-sets
        :return: A list of list containing the minimum k edge cut-sets
        """
        cut_sets = list()
        test = dict()  # Only for debugging tests

        for u, v in itt.combinations(g, 2):
            cut_set = nx.minimum_edge_cut(g, u, v)
            test[(u, v)] = cut_set
            if len(cut_set) == k:
                k_cutsets = list()
                for element in cut_set:
                    k_cutsets.append(tuple(sorted(element)))
                cut_sets.append(sorted(k_cutsets))
        cut_sets.sort()
        return list(element for element, _ in itt.groupby(cut_sets))

    @staticmethod
    def hamilton_path(g):
        """
        Finds a hamiltonian path using networkx graph library, with a backtrack solution
        Not mine. Ref: https://gist.github.com/mikkelam/ab7966e7ab1c441f947b
        :param g: Networkx graph
        :return: Hamiltonian path
        """
        f = [(g, [list(g.nodes())[0]])]  # Fixed
        n = g.number_of_nodes()
        while f:
            graph, path = f.pop()
            confs = []
            for node in graph.neighbors(path[-1]):
                conf_p = path[:]
                conf_p.append(node)
                conf_g = nx.Graph(graph)
                conf_g.remove_node(path[-1])
                confs.append((conf_g, conf_p))
            for g, p in confs:
                if len(p) == n:
                    return p
                else:
                    f.append((g, p))
        return None

    @staticmethod
    def hamilton_cycle(g):
        """
        Finds a hamiltonian cycle using networkx graph library
        :param g: Networkx graph
        :return: Hamiltonian cycle or None if not found
        """
        directed_g = nx.DiGraph(g)
        for cycle in list(nx.simple_cycles(directed_g)):
            if len(cycle) == len(directed_g.nodes):
                return cycle

        return None

    @staticmethod
    def generate_random_treecyc_graphs(path, n_graphs):
        """
        Method that generates random Tree Networkx Graphs.
        :param path: Path to write the files containing the graphs
        :param n_graphs: Maximum number of nodes of the Tree
        :return: A file for each graph containing an Adjacency Matrix
        """
        for i in range(1, n_graphs + 1):
            tree_nodes = i
            if i < 6:
                max_cycle_nodes = 3
            else:
                max_cycle_nodes = int(i / 2)

            r_graph = GraphTools.get_random_treecyc(tree_nodes, max_cycle_nodes)
            graph_id = 'r_treecyc_' + str(i)
            AdjMaBox.write_matrix(r_graph, path + graph_id)

    @staticmethod
    def data_analysis(graph, binomial_format=False, fast=False):
        p = sympy.symbols("p")
        ham_cycle = GraphTools.hamilton_cycle(graph)
        g6_bytes = nx.to_graph6_bytes(graph, nodes=None, header=False)
        # Remove possible end newline
        g6_bytes = g6_bytes.decode().rstrip('\n').encode()

        if fast:
            d = {'g6_hash': hashlib.md5(g6_bytes).hexdigest(),
                 'g6': nx.to_graph6_bytes(graph, nodes=None, header=False),
                 'nodes': str(len(graph.nodes)),
                 'edges': str(len(graph.edges)),
                 'hamiltonian': False if ham_cycle is None else True,
                 'hamiltonian_cycle': ham_cycle,
                 'graph_edges': sorted(graph.edges()),
                 'spanning_trees': GraphTools.spanning_trees_count(graph),
                 'edge_connectivity': nx.edge_connectivity(graph),
                 'min_k2_edge_cut': len(GraphTools.minimum_k_edges_cutsets(graph, 2)),
                 'automorphisms': GraphTools.automorphism_group_number(graph),
                 'diameter': nx.diameter(graph)}

        else:

            poly = GraphRel.relpoly_binary_improved(graph, 0)

            bin_poly = None
            if binomial_format:
                try:
                    bin_poly, bin_coefficients = Utilities.polynomial2binomial(poly)
                except:
                    bin_poly, bin_coefficients = Utilities.polynomial2binomial(poly)
                    bin_poly = 0
                    print(g6_bytes.decode())

            d = {'g6_hash': hashlib.md5(g6_bytes).hexdigest(),
                 'g6': g6_bytes.decode(),
                 'nodes': str(len(graph.nodes)),
                 'edges': str(len(graph.edges)),
                 'hamiltonian': False if ham_cycle is None else True,
                 'hamiltonian_cycle': str(ham_cycle),
                 'graph_edges': str(sorted(graph.edges())),
                 'avg_polynomial': sympy.integrate(poly.as_expr(), (p, 0, 1)),
                 'polynomial': str(bin_poly) if binomial_format else str(poly),
                 'spanning_trees': GraphTools.spanning_trees_count(graph),
                 'edge_connectivity': nx.edge_connectivity(graph),
                 'min_k2_edge_cuts': len(GraphTools.minimum_k_edges_cutsets(graph, 2)),
                 'automorphisms': GraphTools.automorphism_group_number(graph),
                 'diameter': nx.diameter(graph),
                 'probability_01': poly.subs({p: 0.1}),
                 'probability_02': poly.subs({p: 0.2}),
                 'probability_03': poly.subs({p: 0.3}),
                 'probability_04': poly.subs({p: 0.4}),
                 'probability_05': poly.subs({p: 0.5}),
                 'probability_06': poly.subs({p: 0.6}),
                 'probability_07': poly.subs({p: 0.7}),
                 'probability_08': poly.subs({p: 0.8}),
                 'probability_09': poly.subs({p: 0.9})}

        df = pd.DataFrame(data=d, index=[0])
        df.set_index('g6_hash', inplace=True)
        return df

    @staticmethod
    def data_print(data, write_fomat, path):

        if write_fomat == FormatType.CSV:
            data.to_csv(path + ".csv")

        elif write_fomat == FormatType.JSON:
            data.to_json(path + ".json")

        elif write_fomat == FormatType.HTML:
            data.to_html(path + ".html")

        elif write_fomat == FormatType.Excel:
            data.to_excel(path + ".xml")

        elif write_fomat == FormatType.SQL:
            engine = db.create_engine('sqlite:///' + path + ".db", echo=False)
            Session = db.orm.session.sessionmaker(bind=engine)
            session = Session()
            DButilities.add_or_update(session, data, Table_Graph)
            session.close()

    @staticmethod
    def data_read(read_format, path):

        if read_format == FormatType.CSV:
            return pd.read_csv(path)

        elif read_format == FormatType.JSON:
            return pd.read_json(path)

        elif read_format == FormatType.HTML:
            return pd.read_html(path)

        elif read_format == FormatType.Excel:
            return pd.read_excel(path)

        elif read_format == FormatType.SQL:
            return pd.read_sql(path)

    @classmethod
    def __analytics_header(cls, check_super_graph=False, check_coefficients=False):
        """
        It gives the header for witting the results in a file for the 'analyze_graphs' method
        :param check_super_graph: write the header for the column "Super Graph"
        :param check_coefficients: write the headers for the column "Binomial"
        :return: a string with the header columns for the 'analyze_graphs' method
        """
        return ("\n Graph; " +
                ("Super Graph; " if check_super_graph else "") +
                "Hamiltonian; "
                "Hamiltonian cycle; "
                "Graph Edges; "
                "Avg. polynomial; "
                "Polynomial; " +
                ("Binomial; " if check_coefficients else "") +
                "Spanning Trees; "
                "Edge connectivity; "
                "Min. k=2 edge-cut; "
                "Automorphisms; "
                # "Root Number; "
                "Diameter; "
                "Probability 0.1; "
                "Probability 0.2; "
                "Probability 0.3; "
                "Probability 0.4; "
                "Probability 0.5; "
                "Probability 0.6; "
                "Probability 0.7; "
                "Probability 0.8; "
                "Probability 0.9; ")
        # ("Greater coeffs.?; " if check_coefficients else ""))
        # "Roots")

    @classmethod
    def __analytics_body(cls, graph, l_prev_graph=None, check_coefficients=False):
        """
        This method extracts the following information from the given graph:
            - If its a subgraph from one graph of the l_prev_graph list
            - If its hamiltonian
            - Its hamiltonian cycle
            - Its edges
            - The average polynomial
            - Its Rel(G,p) polynomial
            - The binomial form of its Rel(G,p) polynomial
            - The number of spanning trees
            - The edge connectivity
            - The minimum k=2 edge-cut set
            - Number of automorphisms
            - Its diameter
            - The probabilities if p=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        :param graph: graph to analyze
        :param l_prev_graph: optional - list of graphs to check isomorphism
        :param check_coefficients: if 'True' it will show the binomial form of the Rel(G,p) polynomial
        :return: Rel(G,p) polynomial and a string with all the analyzed information
        """
        p = sympy.symbols("p")

        # poly = GraphRel.relpoly_binary_basic(graph)
        poly = GraphRel.relpoly_binary_improved(graph, 0)
        # poly_roots = roots(poly)
        ham_cycle = GraphTools.hamilton_cycle(graph)

        is_sub_graph = False
        if type(l_prev_graph) is list:
            for prev_graph in l_prev_graph:
                if GraphTools.check_one_edge_less_subgraph_isomorphism(graph, prev_graph):
                    is_sub_graph = True
                    break

        bin_poly = None
        if check_coefficients:
            bin_poly, bin_coefficients = Utilities.polynomial2binomial(poly)

        return poly, \
               ("\n" + "Graph_n" + str(len(graph.nodes)) + "_e" + str(len(graph.edges)) + ";" +
                ((str(is_sub_graph) + ";") if type(l_prev_graph) is list else "") +
                str(False if ham_cycle is None else True) + ";" +
                str(ham_cycle) + ";" +
                str(sorted(graph.edges())) + ";" +
                str(sympy.integrate(poly.as_expr(), (p, 0, 1))) + ";" +
                str(poly) + ";" +
                ((str(bin_poly) + ";") if check_coefficients else "") +
                str(GraphTools.spanning_trees_count(graph)) + ";" +
                str(nx.edge_connectivity(graph)) + ";" +
                str(len(GraphTools.minimum_k_edges_cutsets(graph, 2))) + ";" +
                str(GraphTools.automorphism_group_number(graph)) + ";" +
                # str(len(poly_roots)) + ";" +
                str(nx.diameter(graph)) + ";" +
                str(poly.subs({p: 0.1})) + ";" +
                str(poly.subs({p: 0.2})) + ";" +
                str(poly.subs({p: 0.3})) + ";" +
                str(poly.subs({p: 0.4})) + ";" +
                str(poly.subs({p: 0.5})) + ";" +
                str(poly.subs({p: 0.6})) + ";" +
                str(poly.subs({p: 0.7})) + ";" +
                str(poly.subs({p: 0.8})) + ";" +
                str(poly.subs({p: 0.9})))  # + ";" +
        # str(poly_roots))

    @classmethod
    def __analytics_header_fast(cls):
        """
        It gives the header for witting the fast results in a file for the 'analyze_graphs' method
        :return: a string with the header columns for the 'analyze_graphs' method with the 'faster' option
        """
        return ("\n Graph; " +
                "Hamiltonian; "
                "Hamiltonian cycle; "
                "Graph Edges; "
                "Spanning Trees; "
                "Edge connectivity; "
                "Min. k=2 edge-cut; "
                "Automorphisms; "
                "Diameter; ")

    @classmethod
    def __analytics_body_fast(cls, graph):
        """
        This method extracts the following information from the given graph:
            - If its hamiltonian
            - Its hamiltonian cycle
            - Its edges
            - The number of spanning trees
            - The edge connectivity
            - The minimum k=2 edge-cut set
            - Number of automorphisms
            - Its diameter
        :param graph: graph to analyze
        :return: a string with all the analyzed information
        """
        ham_cycle = GraphTools.hamilton_cycle(graph)

        return ("\n" + "Graph_n" + str(len(graph.nodes)) + "_e" + str(len(graph.edges)) + ";" +
                str(False if ham_cycle is None else True) + ";" +
                str(ham_cycle) + ";" +
                str(sorted(graph.edges())) + ";" +
                str(GraphTools.spanning_trees_count(graph)) + ";" +
                str(nx.edge_connectivity(graph)) + ";" +
                str(len(GraphTools.minimum_k_edges_cutsets(graph, 2))) + ";" +
                str(GraphTools.automorphism_group_number(graph)) + ";" +
                str(nx.diameter(graph)))

    @staticmethod
    def analyze_graphs_fast(g_list, path, file_name):
        """
        This method do a fast analysis with basic information of the given graphs and write it into a .txt file,
        in a format prepared to be imported to a calc file.
        :param g_list: list of graphs to analyze
        :param path: path where the file will be written
        :param file_name: name of the file to write
        """
        if os.path.isfile(path + "/Results/" + file_name + "_results_fast.txt"):
            os.remove(path + "/Results/" + file_name + "_results_fast.txt")

        print(GraphTools.__analytics_header_fast(),
              file=open(path + "/Results/" + file_name + "_results_fast.txt", "a"))

        print("Starting analysis...")
        for graph in g_list:
            body = GraphTools.__analytics_body_fast(graph)

            print(body, file=open(path + "/Results/" + file_name + "_results_fast.txt", "a"))

    @staticmethod
    def analyze_graphs(g_list, path, file_name, cross_points=True):
        """
        This method analyzes the given graphs and write the information into a .txt file,
        in a format prepared to be imported to a calc file.
        :param g_list: list of graphs to analyze
        :param path: path where the file will be written
        :param file_name: name of the file to write
        :param cross_points: flag to check if some of the graph polynomials has a cross point with another
        """
        polynomials = list()

        if os.path.isfile(path + "/Results/" + file_name + "_results.txt"):
            os.remove(path + "/Results/" + file_name + "_results.txt")

        print(GraphTools.__analytics_header(), file=open(path + "/Results/" + file_name + "_results.txt", "a"))

        print("Starting analysis...")
        for graph in g_list:
            poly, body = GraphTools.__analytics_body(graph)

            print(body, file=open(path + "/Results/" + file_name + "_results.txt", "a"))

            polynomials.append(poly)

        # Check if the computed polynomials cros in some point
        if cross_points:
            if os.path.isfile(path + "/Results/" + file_name + "_subs.txt"):
                os.remove(path + "/Results/" + file_name + "_subs.txt")

            results = GraphTools.check_polynomial_cross_points(polynomials)

            print("\n Polynomial 1 ; Polynomial 2 ; Subtraction ; Square root ; (0,1) root?; ",
                  file=open(path + "/Results/" + file_name + "_subs.txt", "a"))
            for pol1, pol2, result, square in results:
                keys = square.keys()
                bad = False
                for key in keys:
                    # if key is not sympy.Add:
                    if not isinstance(key, sympy.Add):
                        if 0 < key < 1:
                            bad = True
                if bad:
                    print("\n", pol1, " ;", pol2, " ;", result, " ;", square, "; Cross ;",
                          file=open(path + "/Results/" + file_name + "_subs.txt", "a"))
                else:
                    print("\n", pol1, " ;", pol2, " ;", result, " ;", square,
                          file=open(path + "/Results/" + file_name + "_subs.txt", "a"))

        print("\nDone")

    @staticmethod
    def reliability_polynomial_optimal_graphs(n_nodes, complete):
        """
        Generates optimal graphs for 'n' nodes and up to n/2 edges or complete if indicated. After the graphs
        are generated, then they are analyzed. The results will be written into a .txt file in a format prepared
        to be imported to a calc file.
        :param n_nodes: number of nodes for the graphs
        :param complete: indicates if the number of edges will be if 'True' up to complete or n/2 if 'False'
        """
        file_name = "Graph_n" + str(n_nodes)
        in_path = os.getcwd() + "/Data/Graph6/"
        out_path = os.getcwd() + "/Data/Results/"

        # Complete graph
        if complete:
            edges = int(n_nodes * (n_nodes - 1) / 2)

        # Gives -> nodes/2 <- chords
        else:
            edges = int(n_nodes / 2 + n_nodes)

        if os.path.isfile(out_path + "Opt_Hamiltonian_" + file_name + ".txt"):
            os.remove(out_path + "Opt_Hamiltonian_" + file_name + ".txt")

        print(GraphTools.__analytics_header(True), file=open(out_path + "Opt_Hamiltonian_" + file_name + ".txt", "a"))

        if os.path.isfile(out_path + "Opt_General_" + file_name + ".txt"):
            os.remove(out_path + "Opt_General_" + file_name + ".txt")

        print(GraphTools.__analytics_header(True), file=open(out_path + "Opt_General_" + file_name + ".txt", "a"))

        l_prev_Hgraph = list()
        l_prev_Ggraph = list()
        for i in range(n_nodes, edges + 1):
            print("\nAnalyzing graph:", file_name + "_e" + str(i))

            if os.path.isfile(in_path + file_name + "_e" + str(i) + ".g6"):
                g_list = nx.read_graph6(in_path + file_name + "_e" + str(i) + ".g6")

                # If the read graphs is only one, wrap it with a list
                if type(g_list) is not list:
                    g_list = [g_list]

                filtered_ham, filtered_others = GraphTools.filter_hamiltonian_general_graphs(g_list)

                if len(filtered_ham) > 1:
                    print("WARNING: The filtered Hamiltonian graphs gives more than one optimal")

                if len(filtered_others) > 1:
                    print("WARNING: The filtered General graphs gives more than one optimal")

                if len(filtered_ham) == 0 and len(filtered_others) > 0:
                    print("None of the graphs are Hamiltonians")

                if len(filtered_ham) > 0 and len(filtered_others) == 0:
                    print("All of the graphs are Hamiltonians")

                aux_list = list()
                for opt_Hgraph in filtered_ham:
                    polyH, bodyH = GraphTools.__analytics_body(opt_Hgraph, l_prev_Hgraph)

                    print(bodyH, file=open(out_path + "Opt_Hamiltonian_" + file_name + ".txt", "a"))

                    aux_list.append(opt_Hgraph)

                l_prev_Hgraph = list()
                l_prev_Hgraph = copy.deepcopy(aux_list)

                aux_list = list()
                for opt_Ggraph in filtered_others:
                    polyG, bodyG = GraphTools.__analytics_body(opt_Ggraph, l_prev_Ggraph)

                    print(bodyG, file=open(out_path + "Opt_General_" + file_name + ".txt", "a"))
                    aux_list.append(opt_Ggraph)

                l_prev_Ggraph = list()
                l_prev_Ggraph = copy.deepcopy(aux_list)

            else:
                print("The file ", file_name, " doesn't exist, please make sure that is in the folder ", in_path)

        print("\nDone")

    @staticmethod
    def compare_graphs(file_name, graph1, graph_bunch, cross_points=True, coefficients=False):
        """
        Method that compare different properties of a graph with a bunch of other graphs. The data of the comparisson
        will be written into a .txt file.
        :param file_name: name of the file which the results will be written
        :param graph1: main graph to compare
        :param graph_bunch: graphs to compare with the main graph
        :param cross_points: check if the polynomials of the graphs crosses another graph
        :param coefficients: show the coefficients of the reliability polynomial
        """
        path = os.getcwd() + "/Data/Results/"

        if os.path.isfile(path + file_name + "_comparison.txt"):
            os.remove(path + file_name + "_comparison.txt")

        if coefficients:
            header = GraphTools.__analytics_header(False, True)
            poly1, body1 = GraphTools.__analytics_body(graph1, None, True)

        else:
            header = GraphTools.__analytics_header()
            poly1, body1 = GraphTools.__analytics_body(graph1)

        print(header, file=open(path + file_name + "_comparison.txt", "a"))
        print(body1, file=open(path + file_name + "_comparison.txt", "a"))

        print(header, file=open(path + file_name + "_comparison.txt", "a"))
        for graph in graph_bunch:

            if coefficients:
                poly2, body2 = GraphTools.__analytics_body(graph, None, True)

                bin_poly1, bin_coeff1 = Utilities.polynomial2binomial(poly1)
                bin_poly2, bin_coeff2 = Utilities.polynomial2binomial(poly2)

                greater_coeff = Utilities.compare_coefficients(bin_coeff1, bin_coeff2)

                body2 = str(body2) + "; False" if greater_coeff else "; True"

            else:
                poly2, body2 = GraphTools.__analytics_body(graph)

            print(body2, file=open(path + file_name + "_comparison.txt", "a"))

        # Check if the computed polynomials cros in some point
        if cross_points:
            if len(graph_bunch) > 1:
                print("In order to compare the cross points the input number of graphs must be 2.")

            else:
                results = GraphTools.check_polynomial_cross_points([poly1, poly2])

                print("\n \n Polynomial 1 ; Polynomial 2 ; Subtraction ; Square root ; (0,1) root?; ",
                      file=open(path + file_name + "_comparison.txt", "a"))
                for pol1, pol2, result, square in results:
                    keys = square.keys()
                    bad = False
                    for key in keys:
                        # if key is not sympy.Add:
                        if not isinstance(key, sympy.Add):
                            if 0 < key < 1:
                                bad = True
                    if bad:
                        print("\n", pol1, " ;", pol2, " ;", result, " ;", square, "; Cross ;",
                              file=open(path + file_name + "_comparison.txt", "a"))
                    else:
                        print("\n", pol1, " ;", pol2, " ;", result, " ;", square,
                              file=open(path + file_name + "_comparison.txt", "a"))
        print("\nDone")

    @staticmethod
    def analyze_g6_graphs(file_name, cross_points, g_list=None, fast=False):
        """
        This method will check different properties of the given graphs. And if its indicated, if there is some
        graph that its reliability polynomial crosses another Rel. of another graph.
        :param file_name: file to read the graphs if they're not given
        :param cross_points: check if the polynomials of the graphs crosses another graph
        :param g_list: list of graphs to be analyzed
        :param fast: faster but with less information analysis
        """
        path = os.getcwd() + "/Data"
        if g_list is None:
            g_list = nx.read_graph6(path + "/Graph6/" + file_name + ".g6")
        # If the read graphs is only one, wrap it with a list
        if type(g_list) is not list:
            g_list = [g_list]

        if fast:
            GraphTools.analyze_graphs_fast(g_list, path, file_name)
        else:
            GraphTools.analyze_graphs(g_list, path, file_name, cross_points)

    @staticmethod
    def analyze_g6_files(n_min, n_max, complete=False, cross_points=True):
        """
        This method will check different properties of graphs from n_min to n_max vertices with n/2 edges or up to
        complete graph if indicated. All the results will be written into a .txt file that could be easily converted to
        a calc sheet.
        :param n_min: minimum nodes
        :param n_max: maximum nodes
        :param complete: boolean that indicates if the number of edges is to complete graph or otherwise n/2
        :param cross_points: check if the polynomials of the graphs crosses another graph
        """
        for nodes in range(n_min, n_max + 1):
            # Complete graph
            if complete:
                edges = int(nodes * (nodes - 1) / 2)

            # Gives -> nodes/2 <- chords
            else:
                edges = int(nodes / 2 + nodes)

            for i in range(nodes, edges + 1):
                file_name = "Graph_n" + str(nodes) + "_e" + str(i)
                if os.path.isfile(os.getcwd() + "/Data/Graph6/" + file_name + ".g6"):
                    print("\nAnalyzing file: ", file_name)
                    GraphTools.analyze_g6_graphs(file_name, cross_points)
                    print("\nAnalyzed")

                else:
                    print("\n File ", file_name, ".g6 not found.")

        print("\nDone")

    @staticmethod
    def g6_files_data_analysis(n_min, n_max, max_chords=0, binomial_format=True, fast=False):
        """
        This method will check different properties of graphs from n_min to n_max vertices with n/2 edges or up to
        complete graph if indicated. All the results will be written into a .txt file that could be easily converted to
        a calc sheet.
        :param n_min: minimum nodes
        :param n_max: maximum nodes
        :param complete: boolean that indicates if the number of edges is to complete graph or otherwise n/2
        :param cross_points: check if the polynomials of the graphs crosses another graph
        """
        path = os.getcwd() + "/Data/Graph6/"

        for nodes in range(n_min, n_max + 1):
            # Complete graph
            if max_chords == 0:
                edges = int(nodes * (nodes - 1) / 2)

            # Gives -> nodes/2 <- chords
            else:
                edges = max_chords + nodes

            data_frames = None
            for i in range(nodes, edges + 1):
                file_name = "Graph_n" + str(nodes) + "_e" + str(i)
                if os.path.isfile(path + file_name + ".g6"):
                    print("\nAnalyzing file: ", file_name)
                    g_list = nx.read_graph6(path + file_name + ".g6")
                    if type(g_list) is not list:
                        g_list = [g_list]
                    #TODO: Improve efficiency
                    for g in g_list:
                        if data_frames is None:
                            data_frames = GraphTools.data_analysis(g, binomial_format, fast)
                        else:
                            data_frames = data_frames.append(GraphTools.data_analysis(g, binomial_format, fast))
                    print("\nAnalyzed")

                else:
                    print("\n File ", file_name, ".g6 not found.")

            #TODO: Maybe input write format
            GraphTools.data_print(data_frames, FormatType.SQL,  os.getcwd() + "/Data/DDBB/" + "Graphs_DB")
        print("\nDone")

    @staticmethod
    def gen_g6_files(n_min, n_max, complete=False):
        """
        Generate graphs from n_min to n_max vertices with n/2 edges or up to complete graph. This graphs will be written
        in Graph6 format into a .g6 file
        :param n_min: minimum nodes
        :param n_max: maximum nodes
        :param complete: boolean that indicates if the number of edges is to complete graph or otherwise n/2
        """
        for nodes in range(n_min, n_max + 1):
            # Complete graph
            if complete:
                edges = int(nodes * (nodes - 1) / 2)

            # Gives -> nodes/2 <- chords
            else:
                edges = int(nodes / 2 + nodes)

            print("Files location: " + os.getcwd() + "/Data/Graph6/\n")

            for i in range(nodes, edges + 1):
                file_name = "Graph_n" + str(nodes) + "_e" + str(i) + ".g6"

                if os.path.isfile(os.getcwd() + "/Data/Graph6/" + file_name):
                    os.remove(os.getcwd() + "/Data/Graph6/" + file_name)

                print("\nCreating file: ", file_name)
                command = "nauty-geng " + str(nodes) + " " + str(i) + ":" + str(
                    i) + " -c > " + os.getcwd() + "/Data/Graph6/" + file_name
                os.system(command)
        print("\nDone")

    @staticmethod
    def gen_g6_file(g_list, file_name):
        """
        Transform the given graphs to a Graph6 format and writte them into a .g6 file
        :param g_list: list of graphs to be converted
        :param file_name: name of the file containing the given graphs in Graph6 format
        """
        decoded = ""
        for graph in g_list:
            g6_string = nx.to_graph6_bytes(graph)
            decoded += g6_string.decode("utf-8")[10:]

        print(decoded, file=open(os.getcwd() + "/Data/Graph6/" + file_name + ".g6", "w"))

    @staticmethod
    def generate_random_hamiltonian(n_nodes, chords, p_new_connection=0.1):
        """
        Generate a random Hamiltonian graph with the nodes and chords provided
        :param n_nodes: Number of nodes of the graph
        :param chords: Number of chords of the graph
        :param p_new_connection: probability to add new edge
        :return: A Hamiltonian graph
        """

        cycle = nx.cycle_graph(n_nodes)
        n_edges = len(cycle.edges)
        new_edges = list()

        #  Not enough edges to form a cycle with all the nodes
        if chords < 0:
            raise ValueError("The number of edges must be at least equal to the number of nodes\n")

        #  More edges than the complete graph
        elif n_edges > (n_nodes * (n_nodes - 1)) / 2:
            raise ValueError("The number of edges exceed the complete graph "
                             "with the given nodes (too much edges)")

        # Just enough edges to form a cycle with all the nodes
        elif chords == 0:
            return cycle

        while chords > 0:
            for node in cycle.nodes():
                # find the other nodes this one is connected to
                connected = [to for (fr, to) in cycle.edges(node)]
                connected.append(node)  # To avoid self loops

                # and find the remainder of nodes, which are candidates for new edges
                unconnected = [n for n in cycle.nodes() if n not in connected]

                # probabilistically add a random edge
                if len(unconnected):  # only try if new edge is possible
                    if random() < p_new_connection:
                        new = choice(unconnected)
                        cycle.add_edge(node, new)
                        new_edges.append((node, new))
                        # book-keeping
                        unconnected.remove(new)
                        connected.append(new)
                        chords -= 1
                if chords == 0:
                    break

        return cycle

    @staticmethod
    def gen_all_hamiltonian(n_nodes, chords):
        """
        Generate all possible Hamiltonian graphs with the given nodes and chords
        :param n_nodes: Number of nodes of the Hamiltonian graphs
        :param chords: Number of chords of the Hamiltonian graphs
        :return: All the possible Hamiltonian graphs
        """
        cycle = nx.cycle_graph(n_nodes)
        n_edges = len(cycle.edges)

        #  Not enough edges to form a cycle with all the nodes
        if chords < 0:
            raise ValueError("The number of edges must be at least equal to the number of nodes\n")

        #  More edges than the complete graph
        elif (n_edges + chords) > (n_nodes * (n_nodes - 1)) / 2:
            raise ValueError("The number of edges exceed the complete graph "
                             "with the given nodes (too much edges)")

        # Just enough edges to form a cycle with all the nodes
        elif chords == 0:
            return cycle

        # Get all possible edges of the graph
        nodes = cycle.nodes()
        raw_elements = set()
        for subset in itt.combinations(nodes, 2):
            raw_elements.add(subset)

        # Get current edges in the cycle
        neighbors = set()
        for node in nodes:
            for neighbor in cycle.neighbors(node):
                neighbors.add((node, neighbor) if node < neighbor else (neighbor, node))

        # Get all non-existent possible edges
        elements = raw_elements - neighbors

        # From all non-existent possible edges, get combinations of 'ch' chords and for each combination
        # create a graph and save it into a list of graphs
        ham_graphs = list()
        for subset in itt.combinations(elements, chords):
            new_graph = copy.deepcopy(cycle)

            new_graph.add_edges_from(list(subset))

            ham_graphs.append(new_graph)

        return ham_graphs

    @staticmethod
    def gen_all_3ch_hamiltonian_opt(n_nodes, out_path=None):
        """
        Generate all possible Hamiltonian graphs with the given nodes and chords. The method is Optimized,
        this means, compute the first chord (diagonal), chords of length more than n/2 and maximum grade = 3
        :param n_nodes: Number of nodes of the Hamiltonian graphs
        :param out_path: Optional, path for writing the generated graphs in g6 format
        :return: All the possible Hamiltonian graphs or if out_path is given, a file containing a g6
        format of the graphs
        """
        cycle = nx.cycle_graph(n_nodes)

        # Get the nodes of the graph
        nodes = list(cycle.nodes())

        # Set the diametrical chord
        d1 = nodes[0]
        d2 = nodes[int(len(nodes) / 2)]
        cycle.add_edge(d1, d2)

        # Remove the nodes of the added chord from the nodes list
        nodes.remove(0)
        nodes.remove(int(n_nodes / 2))
        nodes_p1 = nodes[:int(len(nodes) / 2)]
        nodes_p2 = nodes[int(len(nodes) / 2):]

        # Get possible chords
        possible_edges = list()
        for node1 in nodes_p1:
            tmp = list()
            for node2 in nodes_p2:
                tmp.append((node1, node2))
            possible_edges.append(tmp)

        # From all non-existent possible edges, get combinations of 2 chords (since we already have 1) and for each
        # combination create a graph and save it into a list of graphs
        ham_graphs = list()
        for i in range(1, len(possible_edges)):
            sublist1 = possible_edges[i - 1]
            for edge1 in sublist1:
                for sublist2 in possible_edges[i:]:
                    for edge2 in sublist2:
                        new_graph = copy.deepcopy(cycle)

                        new_graph.add_edges_from([edge1, edge2])

                        ham_graphs.append(new_graph)

        if out_path is None:
            return GraphTools.filter_isomorphisms(ham_graphs)
        else:
            GraphTools.filter_isomorphisms(ham_graphs, out_path)

    @staticmethod
    def filter_uniformly_most_reliable_graph(g_list):
        """
        Filters the most reliable graph from a list of graphs by checking the biggest number of spanning trees
        with the greater edge connectivity and the lowest minimum k edges cut-set.
        :param g_list: list of graphs to be filtered
        :return: most reliable graph in the given list
        """

        # Filter the graphs by spanning trees number
        n_sp_filtered = defaultdict(list)
        greater = 0
        for graph in g_list:
            n_sp_trees = GraphTools.spanning_trees_count(graph)
            n_sp_filtered[n_sp_trees].append(graph)

            if greater < n_sp_trees:
                greater = n_sp_trees

        t_optimal = n_sp_filtered[greater]  # Get graphs with the greater spanning trees number

        # If the filter gives only one graph, no need to apply more filters
        if len(t_optimal) == 1:
            return t_optimal

        # Filter the t_optimal graphs by edge connectivity
        edg_con_filtered = defaultdict(list)
        greater = 0
        for graph in t_optimal:
            connectivity = nx.edge_connectivity(graph)
            edg_con_filtered[connectivity].append(graph)

            if greater < connectivity:
                greater = connectivity

        edg_optimal = edg_con_filtered[greater]  # Get graphs with the greater connectivity

        # If the filter gives only one graph, no need to apply more filters
        if len(edg_optimal) == 1:
            return edg_optimal

        # Filter by minimum k edges cut-set
        m_k_filtered = defaultdict(list)
        lowest = -1
        for graph in edg_optimal:
            m_k = len(GraphTools.minimum_k_edges_cutsets(graph, 2))
            m_k_filtered[m_k].append(graph)

            if lowest == -1:
                lowest = m_k

            elif m_k < lowest:
                lowest = m_k

        return m_k_filtered[lowest]

    @staticmethod
    def filter_hamiltonian_general_graphs(g_list, filter_most_reliable=True):
        """
        Filters the hamiltonian graphs from a given list.
        :param g_list: list of graphs to be filtered
        :param filter_most_reliable: flag to filter separately the most reliable general graph and the most
        reliable hamiltonian one
        :return: two lists, one containing the hamiltonian graphs and the other the non-hamiltonian. If the
        flag is in 'True', then will return the most reliable general graph and the most reliable hamiltonian
        """
        hamiltonians = list()
        others = list()
        for graph in g_list:
            ham_cycle = GraphTools.hamilton_cycle(graph)
            if ham_cycle:
                hamiltonians.append(graph)
            else:
                others.append(graph)

        if filter_most_reliable:
            filtered_ham = GraphTools.filter_uniformly_most_reliable_graph(hamiltonians)
            filtered_others = GraphTools.filter_uniformly_most_reliable_graph(others)

            return filtered_ham, filtered_others

        return hamiltonians, others

    @staticmethod
    def filter_isomorphisms(g_list, out_path=None):
        path = os.getcwd() + "/Data/tmp/"
        file_name = "tmp_data"

        # Create a file containing the graphs in g6 format
        for g in g_list:
            g6 = nx.to_graph6_bytes(g, nodes=None, header=False).decode("utf-8")
            print(g6, file=open(path + file_name + ".g6", "a"), end='')

        print("Total:", len(g_list))

        # Filter by isomorphism
        if out_path is None:
            out_file = path + file_name + "filtered.g6"
            command = "nauty-shortg " + path + file_name + ".g6 " + out_file
            os.system(command)

            # Read the file with the filtered graphs
            if os.path.isfile(path + file_name + "filtered" + ".g6"):
                fg_list = nx.read_graph6(path + file_name + "filtered" + ".g6")
                print("Filtered:", len(fg_list))

            # Remove temporal files
            os.remove(path + file_name + ".g6")
            os.remove(path + file_name + "filtered" + ".g6")
            return fg_list

        else:
            out_file = out_path + "Diametral_n" + str(len(g_list[-1].nodes)) + "_ch3.g6"
            if os.path.isfile(out_file):
                os.remove(out_file)

            command = "nauty-shortg " + path + file_name + ".g6 " + out_file
            os.system(command)
            os.remove(path + file_name + ".g6")

    @staticmethod
    def spanning_trees_count(g):
        """
        Method that count the number of spanning trees of the given graph using the formula:

                    Z(G) = 1/n · prod_{i=1}^{n-1} x_i

        Where:
        G: Graph
        n: Number of eigenvalues of the Laplacian matrix of G
        x_i: The 'i' Eigenvalue

        :param g: Networkx Graph
        :return: Number of spanning trees of the given graph
        """
        laplacian = nx.laplacian_matrix(g)  # Laplacian matrix of the given graph

        e = sorted(np.linalg.eigvals(laplacian.A), reverse=True)  # Sorted eigenvalues of L
        e.pop()  # Take out the last element of the eigenvalues (0)

        # Formula for counting the spanning trees
        n_spanning_trees = 1
        for value in e:
            n_spanning_trees *= value
        n_spanning_trees /= (len(e) + 1)  # + 1 (the last element that we take out previously)

        rounded_spt = round(n_spanning_trees)  # Round the given number of trees
        real_spt = int(rounded_spt)  # Get the real part

        """
        # Detect imaginary numbers and print information
        if isinstance(n_spanning_trees, complex):
            print("Imaginary number: ", n_spanning_trees, ". Rounded: ", rounded_spt,
                  ". Real part: ", real_spt)
        """
        return real_spt

    @staticmethod
    def spanning_trees_fair_cake(g):
        """
        Calculates the reliability polynomial coefficients of the given FCG (Fair Cake Graph, must have 3 chords).
        :param g: FCG with 3 chords
        :return: coefficients of the Rel(G,p) polynomial
        """
        n = len(g)

        # 3ch, 1c
        part1 = n

        # 2ch, 2c
        part2 = 3 * (n / 2) ** 2

        # 1ch, 3c
        part3 = 3 * (2 * (n / 6) ** 2 * n / 3 + 2 * n / 6 * (n / 3) ** 2)

        # 4c
        part4 = (n / 6) ** 4 * (Utilities.number_combination(6, 4) - 3)

        return part1 + part2 + part3 + part4

    @staticmethod
    def get_cycles_edges(g):
        """
        Determine the edges of the inner cycles of the given graph 'g'.
        :param g: Networkx graph.
        :return: <list> the edges of the inner cylces.
        """
        g_nd = nx.Graph(g)  # Make a no multicycle and non directed graph copy
        # cycles_nodes_minimum = nx.minimum_cycle_basis(g_nd)
        cycles_nodes = nx.cycle_basis(g_nd)  # TODO: Check if correct

        n_edges = 0
        bal_pedges = True  # Check if the graph has balanced pallel edges
        aux = -1  # Aux variable to check if the graph has balanced edges

        cycles_edges = []
        for cycle in cycles_nodes:
            cycle_edge = {}
            for node in cycle:
                edges = set(g.edges(node))
                for edge in edges:
                    t = edge[1]
                    if t in cycle:
                        edge = tuple(sorted(edge))
                        n_edges = g.number_of_edges(edge[0], edge[1])
                        cycle_edge[edge] = n_edges
                        # Check if g has balanced edges
                        if aux == -1:
                            aux = n_edges
                        if aux != n_edges:
                            bal_pedges = False

            cycles_edges.append(cycle_edge)
        return cycles_edges, (bal_pedges, n_edges)

    @staticmethod
    def get_detailed_graph_edges(g):
        """
        This function diferentiates the edges that are part of the 'tree' of the graph
        and the edges that are part of a cycle
        :param g - Networkx Graph
        :return:
            bal_pedges - <tuple> If balanced parallel edges, <True, number_of_parallel_edges>,
                            otherwise <False, -1>
            tree_edges - <map> 'tree' edges part of the graph <key:edge, value:n_edges>
            cycle_edges - <list> of maps <key:edge, value:n_edges> where each map contains the information
            of the edges of one cycle
        """
        total_edges = {}  # Dictionary

        #  Get the edges of the minimum cycles of the graph
        cycles_edges, bal_pedges = GraphTools.get_cycles_edges(g)

        # Sort edges
        graph_edges = set()
        for edge in g.edges():
            graph_edges.add(tuple(sorted(edge)))

        total_cycles_edges = set()
        for cycle in cycles_edges:
            total_cycles_edges.update(set(cycle.keys()))
            total_edges.update(cycle)

        # Get the edges of the tree part of the graph
        tree_edges_set = graph_edges - total_cycles_edges
        tree_edges = {}  # map
        # Aux variables to check if the graph has balanced edges
        aux_bool = bal_pedges[0]
        aux_num = bal_pedges[1]

        for edge in tree_edges_set:
            n_edges = g.number_of_edges(edge[0], edge[1])
            tree_edges[edge] = n_edges
            total_edges.update(tree_edges)
            # Check if g has balanced edges
            if aux_num == -1:
                aux_num = n_edges
            if aux_num != n_edges:
                aux_bool = False

        return (aux_bool, aux_num) if aux_bool else (aux_bool, -1), tree_edges, cycles_edges, total_edges

    @staticmethod
    def get_a_common_edge(g):
        """
        Get a common edge between two cycles if exists
        :param g: Networkx graph
        :return: common_edge - common edge <set> or None if doesn't exist any
        """
        h = nx.Graph(g)
        cycles = nx.minimum_cycle_basis(h)  # cycle_basis return a nearest ordered list of cycles

        while len(cycles) > 1:
            cycle1 = set(cycles.pop())

            # Start from last to first element of the list (just for a bit optimization)
            for cycle2 in reversed(cycles):
                cycle2 = set(cycle2)
                common_edge = cycle1.intersection(cycle2)

                # If we have only two nodes, means that is the shared edge
                if common_edge and len(common_edge) == 2:
                    return common_edge

                # If we have more than two nodes, then get the shared edge of the resulting intersection
                elif len(common_edge) > 2:
                    # Get the smallest cycle
                    if len(cycle1) < len(cycle2):
                        temp = cycle1
                    else:
                        temp = cycle2

                    shortest_common_edges = temp - common_edge
                    for node in shortest_common_edges:
                        edge_lst = g.edges(node)  # Get the edges of the node
                        for edge in edge_lst:
                            # Select an edge part of the cycle
                            if set(edge) - temp == set():
                                return edge

        # In case it doesn't has any common edges
        return None

    @staticmethod
    def get_random_treecyc(tn, cn_max):
        """
        Gives a random Netwokx Tree Graph
        :param tn: Tree Nodes
        :param cn_max: Must be at least 3
        :return: Random Netwokx Tree Graph
        """
        tree = nx.random_tree(tn)
        cp_tree = copy.deepcopy(tree)

        r_int = randint(1, tn)

        r_nodes = sample(range(0, tn), r_int)  # Get a list of random nodes

        # Prepare graph for incision
        for node in r_nodes:
            tree = copy.deepcopy(cp_tree)  # Prevent exceptions if the nodes are modified
            tree_len = len(cp_tree.nodes)
            node_edges = tree.edges(node)
            deleted_edges = list()
            new_cycle = list()
            for edge in node_edges:
                # Aislate the node by deleting its edges
                deleted_edges.append(edge[1])
                cp_tree.remove_edge(edge[0], edge[1])

            r_int = randint(3, cn_max)
            prev_node = node
            for new_node in range(tree_len, tree_len - 1 + r_int):
                cp_tree.add_edge(prev_node, new_node)
                new_cycle.append(prev_node)
                prev_node = new_node

            cp_tree.add_edge(prev_node, node)
            new_cycle.append(prev_node)
            for del_node in deleted_edges:
                ch = choice(new_cycle)
                cp_tree.add_edge(del_node, ch)

        return cp_tree

    @staticmethod
    def get_nodes_from_edges(edges):
        """
        Given a list of edges, returns its nodes
        :param edges: list of edges
        :return: list of nodes belonging to the given edges
        """
        nodes = set()
        for edge in edges:
            for node in edge:
                nodes.add(node)
        return list(nodes)

    @staticmethod
    def get_all_connected_graphs(g):
        """
        For each subset of k disconnected edges, gives all the connected graphs.
        :param g: Networkx graph
        :return: Dictionary where <key>: number of deleted edges, <value> list of connected graphs
        """
        connected_graphs = {}
        l_edges = list(g.edges())
        dsc = 1  # number of edge disconnections

        while dsc < len(l_edges):
            aux_l = list()
            for subset in itt.combinations(l_edges, dsc):
                tmp_g = copy.deepcopy(g)
                tmp_g.remove_edges_from(subset)

                if nx.is_connected(tmp_g):
                    aux_l.append(tmp_g)

            if aux_l == list():
                return connected_graphs

            connected_graphs[dsc] = aux_l
            dsc += 1

    @staticmethod
    def differentiate_ordered_cycles(cycles):
        """
        Differentiate the mixed cycles by the following rules:
                - For each set of shared edges -> Only two vertices of the set can have a degree > 2.
                - Any node of the graph could have degree > 3
        :param cycles: list of Networkx cycle graphs
        :return: a tuple with the list of 'ordered' cycles and a list of a 'other' cycles
        """
        other_graphs = list()
        ordered_cycles = list()
        for sg in cycles:

            tmp_bal_pedges, tmp_tree_edges, tmp_cycle_edges, tmp_total_edges = GraphTools.get_detailed_graph_edges(sg)

            # If the subgraph isn't multiedge, analyze it
            if tmp_bal_pedges[0] is True and tmp_bal_pedges[1] == 1:
                shared_edges = list()
                other = False

                # Check the degrees of the graph
                degrees = sg.degree(list(sg.nodes))
                for degree in degrees:
                    # If the degreee is more than 3 the sub-graph then save to 'others' list
                    if degree[1] > 3:
                        other_graphs.append(sg)
                        other = True
                        break
                if other:
                    break

                tmp_sg = copy.deepcopy(sg)

                aux_cycle_edges = copy.deepcopy(tmp_cycle_edges)

                # Get the shared cycle edges
                while aux_cycle_edges:
                    cycle1 = aux_cycle_edges.pop()
                    for cycle2 in aux_cycle_edges:
                        shared = Utilities.intersection(cycle1, cycle2)
                        if shared != list():
                            shared_edges.append(shared)

                # Flatten edge_lst
                flt_edges = Utilities.flatten_nested_list(shared_edges)

                """
                After removing the sharing edges we will have a graph with isolated vertices, 
                this ones will be the inner vertices
                """
                tmp_sg.remove_edges_from(flt_edges)

                # Check the degrees of the inner 'shared' nodes
                degrees = sg.degree(list(nx.isolates(tmp_sg)))
                for degree in degrees:
                    # If the degreee is more than 2 the sub-graph then save to 'others' list
                    if degree[1] > 2:
                        other_graphs.append(sg)
                        other = True
                        break

                if not other:
                    ordered_cycles.append(sg)

            # If the graph is Multi-Edge, then save it to 'others' graph list
            else:
                other_graphs.append(sg)

        return ordered_cycles, other_graphs

    # TODO: Optimize algorithm time
    @staticmethod
    def get_sub_graphs(g, filter_ordered_cycles):
        """
        Method that will 'split' the given graph with its tree and cycle parts. The other sub-graphs will be given
        as a 'merged' sub-graphs
        :param g: networkx graph
        :param filter_ordered_cycles: boolean that determinate if the function filter the ordered cycles or not. If not
        the graphs are marked as "others"
        :return: Edges of the tree part, list of pure cycles, list of merged cycles
        """
        all_subgraphs = list()

        bal_pedges, tree_edges, cycle_edges, total_edges = GraphTools.get_detailed_graph_edges(g)
        pure_cycles = list()
        ordered_cycles = list()
        other_graphs = list()

        g_copy = copy.deepcopy(g)

        if tree_edges != {}:
            all_subgraphs.append((GraphType.Tree, tree_edges))

        # -------------------------------Get the tree sub-graph of the given graph--------------------------------------
        aux_tree_edges = list()
        for edge in tree_edges:
            for i in range(0, tree_edges[edge]):
                aux_tree_edges.append((edge[0], edge[1], i))

        sub_tree = g_copy.edge_subgraph(aux_tree_edges)
        # Remove the 'tree' part of the graph
        g_copy.remove_edges_from(list(sub_tree.edges()))

        # ------------------------------Extract biconected components of the graph--------------------------------------
        bi_components = list((g_copy.subgraph(c) for c in nx.biconnected_components(g_copy)))
        while bi_components:
            bi_component = bi_components.pop()

            # Categorize the subgraphs in pure, ordered cycles or 'others'
            sub_cycles = nx.minimum_cycle_basis(nx.Graph(bi_component))
            # Pure cycles
            if len(sub_cycles) == 1:
                pure_cycles.append(copy.deepcopy(bi_component))

            else:
                # Differentiate the mixed cycles into ordered or 'others
                disconnected_subgraphs = list(nx.connected_component_subgraphs(bi_component))

                if filter_ordered_cycles:
                    tmp_ordered_cycles, tmp_other_graphs = GraphTools.differentiate_ordered_cycles(
                        disconnected_subgraphs)

                else:
                    tmp_ordered_cycles = list()
                    tmp_other_graphs = disconnected_subgraphs

                if tmp_ordered_cycles:
                    ordered_cycles += tmp_ordered_cycles

                if tmp_other_graphs:
                    other_graphs += tmp_other_graphs

            g_copy.remove_edges_from(list(bi_component.edges()))  # Delete bicomponents from the main graph

        # Remove isolated nodes
        g_copy.remove_nodes_from(list(nx.isolates(g_copy)))

        # ------------------------------------Categorize the cycles of the main graph---------------------------------
        disconnected_subgraphs = nx.connected_component_subgraphs(g_copy)  # (g_copy, True) to copy the graph attributes
        for component in disconnected_subgraphs:
            sub_cycles = nx.minimum_cycle_basis(nx.Graph(component))
            if len(sub_cycles) == 1:
                pure_cycles.append(component)
                # Remove the pure cycle edges from the graph (that will create isolated nodes)
                g_copy.remove_edges_from(list(component.edges()))

            else:
                # Differentiate the mixed cycles
                disconnected_subgraphs = list(nx.connected_component_subgraphs(component))

                tmp_ordered_cycles, tmp_other_graphs = GraphTools.differentiate_ordered_cycles(disconnected_subgraphs)

                if tmp_ordered_cycles:
                    ordered_cycles += tmp_ordered_cycles

                if tmp_other_graphs:
                    other_graphs += tmp_other_graphs

        if pure_cycles != list():
            all_subgraphs.append((GraphType.PureCycle, pure_cycles))

        # Remove isolated nodes
        g_copy.remove_nodes_from(list(nx.isolates(g_copy)))

        """
        # Differentiate the mixed cycles
        disconnected_subgraphs = list(nx.connected_component_subgraphs(g_copy))

        ordered_cycles, other_graphs = GraphTools.differentiate_ordered_cycles(disconnected_subgraphs)
        """
        if ordered_cycles != list():
            all_subgraphs.append((GraphType.OrderedCycles, ordered_cycles))

        if other_graphs != list():
            all_subgraphs.append((GraphType.Others, other_graphs))

        return all_subgraphs

    @staticmethod
    def automorphism_group_number(g):
        """
        Method that gives the automorphism group number of the given graph
        :param g: networkx graph
        :return: automorphism group number of the given graph
        """
        gm = isomorphism.GraphMatcher(g, g)
        return sum(1 for _ in gm.isomorphisms_iter())

    @staticmethod
    def check_polynomial_cross_points(polynomials):
        """
        This method checks if the given polynomials cross in some point
        :param polynomials: polynomials to check
        :return: crossing points
        """
        results = list()

        for i in range(0, len(polynomials)):
            prev = polynomials[i]

            for j in range(i + 1, len(polynomials)):
                tmp_res = prev - polynomials[j]

                results.append((prev, polynomials[j], tmp_res, roots(tmp_res)))

        return results

    @staticmethod
    def spanning_trees_polynomial(a, b, c, d, e, f):
        """
        Gives the coefficients polynomials of the Rel(G,p) polynomial based on the lengths between chord endpoints.
        :param a: length a
        :param b: length b
        :param c: length c
        :param d: length d
        :param e: length f
        :param f: length e
        :return: coefficient polynomials
        """
        n = sum([a, b, c, d, e, f])

        poly1_p1 = 3

        poly1_p2 = 3 * n

        poly1_p3 = 0
        for subset in itt.combinations([a, b, c, d, e, f], 2):
            aux = 1
            for i in subset:
                aux *= i
            poly1_p3 += aux

        poly1 = poly1_p1 + poly1_p2 + poly1_p3

        poly2_p1 = 1

        poly2_p2 = 3 * n

        poly2_p3 = (e + f) * a + \
                   (e + f) * (b + c) + \
                   (e + f) * d + \
                   a * (b + c) + \
                   a * d + \
                   (b + c) * d + \
 \
                   (f + a) * b + \
                   (f + a) * (c + d) + \
                   (f + a) * e + \
                   b * (c + d) + \
                   b * e + \
                   (c + d) * e + \
 \
                   (a + b) * c + \
                   (a + b) * (d + e) + \
                   (a + b) * f + \
                   c * (d + e) + \
                   c * f + \
                   (d + e) * f

        poly2_p4 = 0

        for subset in itt.combinations([a, b, c, d, e, f], 3):
            aux = 1
            for i in subset:
                aux *= i
            poly2_p4 += aux

        poly2 = poly2_p1 + poly2_p2 + poly2_p3 + poly2_p4

        poly3_p1 = n

        poly3_p2 = (a + b + c) * (d + e + f) + \
                   (b + c + d) * (e + f + a) + \
                   (c + d + e) * (f + a + b)

        poly3_p3 = (a + b) * c * (d + e) + \
                   (a + b) * c * f + \
                   (a + b) * (d + e) * f + \
                   c * (d + e) * f + \
                   (b + c) * d * (e + f) + \
                   (b + c) * d * a + \
                   (b + c) * (e + f) * a + \
                   d * (e + f) * a + \
                   (c + d) * e * (a + f) + \
                   (c + d) * e * b + \
                   (c + d) * (a + f) * b + \
                   e * (a + f) * b

        poly3_p4 = 0
        for subset in itt.combinations([a, b, c, d, e, f], 4):
            aux = 1
            for i in subset:
                aux *= i
            poly3_p4 += aux

        subtract = a * b * d * e + b * c * e * f + c * d * f * a

        poly3 = poly3_p1 + poly3_p2 + poly3_p3 + poly3_p4 - subtract

        return sympy.simplify(poly1), sympy.simplify(poly2), sympy.simplify(poly3)


class DButilities(object):

    @staticmethod
    def read_table(session, table, column='*', conditions=None):
        if conditions:
            query = "SELECT " + column + " FROM " + table.__tablename__ + " WHERE " + conditions

        else:
            query = "SELECT " + column + " FROM " + table.__tablename__

        df = pd.read_sql_query(query, session.get_bind())

        metadata = db.MetaData()
        graphs = db.Table(table.__tablename__, metadata, autoload=True, autoload_with=session.get_bind())
        table_skeleton = {col.name: col.type.python_type for col in graphs.c}

        df = df.astype(table_skeleton)
        df.set_index(db.inspect(table).primary_key[0].name, inplace=True)
        return df

    @staticmethod
    def create_table(engine, table):
        """
        The create_all() function uses the engine object to create all the defined table objects
        and stores the information in metadata.
        :param engine: sqlalchemy engine
        """
        if not engine.dialect.has_table(engine, table.__tablename__):  # If table don't exist, Create.
            table.__table__.create(bind=engine)

    @staticmethod
    def add_column(session, table_name, column_name, column_type):
        session.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, column_name, column_type))

    @staticmethod
    def bulk_insert(session, df, table_class):
        session.bulk_insert_mappings(table_class, df.to_dict(orient="records"))
        session.commit()

    @staticmethod
    def bulk_update(session, df, table_class):
        session.bulk_update_mappings(table_class, df.to_dict(orient="records"))
        session.commit()

    @staticmethod
    def add_or_update(session, df, table_class):
        rows = DButilities.df_to_objects(df)
        primary_key = db.inspect(table_class).primary_key[0].name

        # Create table if not created
        DButilities.create_table(session.get_bind(), table_class)

        data_update = []
        data_insert = []
        for key, values in rows:
            # Returns filter query
            qry = session.query(table_class).filter(getattr(table_class, primary_key) == key)
            # If qry.one(), then returns the filtered object
            tmp = vars(values)
            tmp[primary_key] = key

            # It exists already
            if qry.count():
                # session.merge(element)  # This will update one by one
                data_update.append(vars(values))

            # It doesn't exist yet
            else:
                # session.add(element)  # This will add one by one
                data_insert.append(vars(values))

        # With the filtered data make bulk updates and inserts
        session.bulk_update_mappings(table_class, data_update)
        session.bulk_insert_mappings(table_class, data_insert)
        session.commit()

    @staticmethod
    def df_to_objects(df):
        obj_lst = list()
        for key, values in df.to_dict(orient="index").items():
            obj_lst.append((key.decode("utf-8") if type(key) is bytes else key, types.SimpleNamespace(**values)))
        return obj_lst


class GraphRel(object):

    @staticmethod
    def relpoly_binary_basic(g):
        """
        Provides the reliability polynomial using the basic contraction-deletion algorithm
        :param g: Graph to calculate
        :return: Reliability polynomial
        """
        h = nx.MultiGraph(g)
        rel = GraphRel.__recursive_basic(h)

        #return sympy.simplify(rel)
        return rel

    @classmethod
    def __recursive_basic(cls, g):
        """
        Recursive contraction-deletion algorithm method that calculates the reliability polynomial
        :param g: Graph to calculate
        :return: Reliability polynomial
        """
        p = sympy.symbols('p')

        # If the graph is not connected, then it has a rel poly of 0
        if not nx.is_connected(g):
            return sympy.Poly(0, p)

        # if # edges > 0, then we perform the two subcases of the Factoring theorem.
        if len(g.edges()) > 0:
            e = choice(list(g.edges()))
            contracted = nx.contracted_edge(g, e, self_loops=False)
            g.remove_edge(*e)
            rec_deleted = GraphRel.__recursive_basic(g)
            rec_contracted = GraphRel.__recursive_basic(contracted)
            s = sympy.Poly(p) * rec_contracted + sympy.Poly(1 - p) * rec_deleted
            return s

        # Otherwise, we only have 0 edges and 1 vertex, which is connected, so we return 1.
        return sympy.Poly(1, p)

    @staticmethod
    def relpoly_binary_improved(g, filter_depth=0):
        """
        Provides the Reliability Polynomial using an improved contraction-deletion algorithm
        :param g: Graph to calculate
        :param filter_depth: number of subgraphs that will be analyzed to determine if they're ordered cycles
        :return: Reliability polynomial
        """
        rel = GraphRel.__recursive_improved(nx.MultiGraph(g), filter_depth)
        # rel = GraphRel._recursive_improved_old(nx.MultiGraph(g))

        return rel

    @classmethod
    def __recursive_improved(cls, g, filter_depth):
        """
        This is the improved contraction-deletion algorithm. In each recursion, if there exist some method
        that can retrieve the Reliability Polynomial directly or with less cost than another recursion,
        will retrieve it and stop the recursion in that generated sub-graph.
        :param g: Networkx graph
        :param filter_depth: number of subgraphs that will be analyzed to determine if they're ordered cycles
        :return: The reliability Polynomial of the given graph or another execution of the method.
        """
        # print("---------Input graph-----")
        # AdjMaBox.plot(g)
        p = sympy.symbols('p')
        polynomial = 1

        """
        # If the graph is k > 2 regular, proceed with contraction-deletion
        elif nx.is_distance_regular(g) and g.degree(choice(g.nodes())) > 2:
            for other in type[1]:
                # if other type, then we perform the two subcases of the Factoring theorem.
                # Look for joined cycles, to optimize the choosed edge
                # common_edge = GraphTools.get_a_common_edge(other)

                # e = copy.deepcopy(common_edge)
                e = choice(list(other.edges()))

                contracted = nx.contracted_edge(other, e, self_loops=False)  # TODO: Expected tuple

                other.remove_edge(*e)
                # AdjMaBox.plot(other)
                rec_deleted = GraphRel.__recursive_improved(other)
                # AdjMaBox.plot(contracted)

                rec_contracted = GraphRel.__recursive_improved(contracted)

                polynomial *= sympy.Poly(p) * rec_contracted + sympy.Poly(1 - p) * rec_deleted
        """

        # If the graph is not connected, then it has a rel poly of 0
        if not nx.is_connected(g):
            return sympy.Poly(0, p)

        # If we only have 0 edges and 1 vertex, is connected, so we return 1.
        elif len(g.edges()) == 0:
            return sympy.Poly(1, p)

        # Else, separate the graph into subgraphs
        else:
            subgraphs = GraphTools.get_sub_graphs(g, False if filter_depth == 0 else True)

            for g_type in subgraphs:

                if g_type[0] == GraphType.Tree and g_type[1] != {}:
                    # TODO: Check if we can obtain the balancing flag
                    polynomial *= GraphRel.relpoly_multitree((False, -1), g_type[1])

                elif g_type[0] == GraphType.PureCycle and g_type[1] != list():
                    for pure_cycle in g_type[1]:
                        # Get information about the cycle/multicycle
                        mcy_bal_pedges, mcy_tree_edges, mcy_cycle_edges, mcy_total_edges = GraphTools.get_detailed_graph_edges(
                            pure_cycle)
                        polynomial *= GraphRel.relpoly_multicycle(mcy_bal_pedges,
                                                                  mcy_cycle_edges[0])  # TODO: mcy_cycle_edges[0] ?

                elif g_type[0] == GraphType.OrderedCycles and g_type[1] != list():
                    for ordered_cycles in g_type[1]:
                        # Get information about the graph
                        ocy_bal_pedges, ocy_tree_edges, ocy_cycle_edges, ocy_total_edges = \
                            GraphTools.get_detailed_graph_edges(ordered_cycles)
                        polynomial *= GraphRel.relpoly_ordered_cycles(ordered_cycles, ocy_cycle_edges)

                elif g_type[0] == GraphType.Others and g_type[1] != list():
                    if filter_depth > 0:
                        filter_depth -= 1  # Depth control

                    for other in g_type[1]:
                        # if other type, then we perform the two subcases of the Factoring theorem.
                        # Look for joined cycles, to optimize the choosed edge

                        # TODO: The function .get_a_common_edge doesn't work due a maltfunction of the networkx function minimum_cycle_basis
                        # common_edge = GraphTools.get_a_common_edge(other) # Not working for ordered cycles
                        # e = copy.deepcopy(common_edge)

                        # TODO: Needs opitmization
                        e = choice(list(other.edges()))  # Random choice

                        contracted = nx.contracted_edge(other, e, self_loops=False)  # TODO: Expected tuple

                        other.remove_edge(*e)
                        # AdjMaBox.plot(other)
                        rec_deleted = GraphRel.__recursive_improved(other, filter_depth)
                        # AdjMaBox.plot(contracted)

                        rec_contracted = GraphRel.__recursive_improved(contracted, filter_depth)

                        polynomial *= sympy.Poly(p) * rec_contracted + sympy.Poly(1 - p) * rec_deleted

        return polynomial

    @staticmethod
    def relpoly_treecyc(g, cycle_edges):
        """
        Get the Reliability Polynomial of a tree+cycles graph shape (no multiedge).
        :param g: Networkx graph.
        :param cycle_edges: <list> (list), list of the edges of each cycle of the graph.
        :return: Reliability Polynomial
        """
        p = sympy.symbols('p')

        cycles = cycle_edges
        n_edges = len(g.edges)
        polynomial = p ** n_edges

        # Broken edges >1
        for L in range(1, len(cycles) + 1):
            result = 0

            for subset in itt.combinations(cycle_edges, L):

                # If only one broken edge
                if L == 1:
                    oper = 0
                    for cycle in subset:
                        oper += len(cycle)

                else:
                    oper = 1
                    for cycle in subset:
                        oper *= len(cycle)

                result += oper
            n_edges -= 1
            polynomial += result * p ** n_edges * (1 - p) ** L

        """
        # For debug purposes
        s_ref = GraphRel.relpoly_binary_basic(g)

        if sympy.Poly(polynomial) != s_ref:
            print("Error in graph: ", g.edges, "\n\n Improved Rel treecyc:\n", polynomial, polynomial.subs({p: 0.6}),
                  "\n Basic Rel:\n",
                  s_ref, "= ", s_ref.subs({p: 0.6}), "\n\n\n", file=open("deb_output.txt", "a"))
        """

        return sympy.Poly(polynomial)

    @staticmethod
    def relpoly_two_fused_cycles(g, cycle_edges):
        """
        This algorithm calulates the reliability polynomial of a graph that consist of two cycles that share edges,
        It works also if the graph has trees connected to the cycles.
        :param g: Networkx graph.
        :param cycle_edges: <list> (list), list of the edges of each cycle of the graph.
        :return: Reliability Polynomial
        """
        p = sympy.symbols('p')
        n_edges = len(g.edges)

        cycle1 = cycle_edges.pop()
        cycle2 = cycle_edges.pop()

        shared_edges = len(Utilities.intersection(cycle1, cycle2))
        ext_cyc1 = len(cycle1) - shared_edges
        ext_cyc2 = len(cycle2) - shared_edges

        polynomial = \
            p ** n_edges \
            + (ext_cyc1 + ext_cyc2 + shared_edges) * p ** (n_edges - 1) * (1 - p) \
            + (shared_edges * (ext_cyc1 + ext_cyc2) + ext_cyc1 * ext_cyc2) * p ** (n_edges - 2) * (1 - p) ** 2

        return sympy.Poly(polynomial)

    @staticmethod
    def relpoly_ordered_cycles(g, cycle_edges=None):
        """
        Specialized formula to calculate the reliability polynomial of graphs which are ordered cycles
        :param g: ordered cycle graph
        :param cycle_edges: number of edges of the outer cycle
        :return: reliability polynomial
        """
        # TODO: Clean unnecessary code after debugging tests
        p = sympy.symbols('p')

        if cycle_edges is None:
            bal_pedges, tree_edges, cycle_edges, total_edges = GraphTools.get_detailed_graph_edges(g)

        n_edges = len(g.edges)
        graph_edges = g.edges()
        no_shared_edges = copy.deepcopy(graph_edges)
        tmp_cycle_edges = copy.deepcopy(cycle_edges)
        tmp_cycle_edges.reverse()

        shared_edges = list()
        n_shared_edges = list()

        n_clean_cycles = list()
        sh = 0  # Number of shared edges
        # The maximum of edges that can be broken = the number of cycles of the graph
        max_broken_edges = len(cycle_edges)

        # Get the shared cycle edges
        while tmp_cycle_edges:
            cycle1 = tmp_cycle_edges.pop()
            for cycle2 in tmp_cycle_edges:
                shared = Utilities.intersection(cycle1, cycle2)
                if shared != list():
                    shared_edges.append(shared)
                    n_shared_edges.append(len(shared))  # TODO: Maybe no need
                    sh += 1
                    # Get the edges that aren't shared
                    no_shared_edges = Utilities.difference(no_shared_edges, shared)  # TODO: Maybe no need

        cycles_index = list()
        # Get the cycles without the shared edges
        for cycle in cycle_edges:
            cycle = list(cycle.keys())
            tmp_sh = list()
            for sh_edges in shared_edges:
                tmp_cyc = Utilities.difference(cycle, sh_edges)
                if len(tmp_cyc) < len(cycle):
                    tmp_sh.append(sh_edges)
                    cycle = tmp_cyc
            cycles_index.append((tmp_sh, cycle))
            n_clean_cycles.append(len(cycle))

        # Calculate the lengths of the cycles depending on which shared edges sets are removed
        internal_comb_index = {}  # list< pair< n_comb_shared_edges, cycle_length > >
        # Edges breaked
        for edge_breaks in range(1, max_broken_edges):
            combinations = list()
            # Possible combinations with the number of shared edges chosen
            for comb in itt.combinations(shared_edges, edge_breaks):

                internal_comb = 0

                mult = 1
                for sh_list in comb:
                    mult *= len(sh_list)
                internal_comb += mult

                tmp_g = nx.Graph(copy.deepcopy(g))

                for edges_lst in comb:
                    tmp_g.remove_edges_from(edges_lst)

                # Get information about the graph without the 'cut' edges
                sub_cycle_edges = GraphTools.get_cycles_edges(tmp_g)

                # Get the cycles without the shared edges
                n_clean_subcycles = list()
                for sub_cycle in sub_cycle_edges[0]:
                    sub_cycle = list(sub_cycle.keys())
                    tmp_sh = list()
                    for sh_edges in shared_edges:
                        tmp_cyc = Utilities.difference(sub_cycle, sh_edges)
                        if len(tmp_cyc) < len(sub_cycle):
                            tmp_sh.append(sh_edges)
                            sub_cycle = tmp_cyc
                    n_clean_subcycles.append(len(sub_cycle))

                combinations.append((internal_comb, n_clean_subcycles))

            internal_comb_index[edge_breaks] = combinations

        polynomial = \
            p ** n_edges \
            + n_edges * p ** (n_edges - 1) * (1 - p)

        # Depending the broken edges (exponent), the component will be different
        # TODO: Check if there aren't any internal edges in the graph
        for edge_breaks in range(2, max_broken_edges + 1):
            constant = 0
            # Calculate the number of external paths
            for comb in itt.combinations(n_clean_cycles, edge_breaks):
                mult = 1
                for num in comb:
                    mult *= num
                constant += mult

            # Calculate the number of internal paths
            for sh_choices in range(1, edge_breaks + 1):

                # If reached the maximum breaks, return the polynomial
                if sh_choices == max_broken_edges:
                    polynomial += constant * p ** (n_edges - edge_breaks) * (1 - p) ** edge_breaks
                    return sympy.Poly(polynomial)

                for internal_chosen in internal_comb_index[sh_choices]:

                    if 1 == sh_choices < edge_breaks == 2:

                        total = 0
                        for cycles_length in internal_chosen[1]:
                            total += cycles_length

                        constant += total * internal_chosen[0]

                    elif sh_choices == edge_breaks:
                        constant += internal_chosen[0]

                    else:
                        for comb in itt.combinations(internal_chosen[1], edge_breaks - sh_choices):
                            total = 1
                            for num in comb:
                                total *= num

                            constant += total * internal_chosen[0]

            polynomial += constant * p ** (n_edges - edge_breaks) * (1 - p) ** edge_breaks

        return sympy.Poly(polynomial)

    @staticmethod
    def relpoly_multitree(bal_pedges, tree_edges):
        """
        Get the polynomial reliability of any tree (multiedge or not)
        :param bal_pedges: <tuple> (bool, int); <bool> if is parallel, <int> number of parallel edges
        :param tree_edges: <list> of the edges of the graph tree
        :return: Reliability polynomial of the graph
        """
        p = sympy.symbols('p')

        # Specialized inclusion-exclusion formula for multitrees with balanced parallel edges
        if bal_pedges[0]:
            return (1 - (1 - p) ** bal_pedges[1]) ** len(tree_edges)

        # Tree Inclusion-Exclusion formula (chunks formula)
        polynomial = 1
        for edgeset in tree_edges:
            polynomial *= 1 - (1 - p) ** tree_edges[edgeset]

        # return sympy.Poly(polynomial)  # TODO: Check if works
        return polynomial

    @staticmethod
    def relpoly_multicycle(bal_pedges, cycle_edges):
        """
        Get the polynomial reliability of any cycle (multiedge or not)
        :param bal_pedges: <tuple> (bool, int); <bool> if is parallel, <int> number of parallel edges
        :param cycle_edges: <list> (list), list of the edges of each cycle of the graph, in this case,
                            only one
        :return: Reliability polynomial of the graph
        """
        p = sympy.symbols('p')
        polynomial = 0

        # Specialized inclusion-exclusion formula for multicycles with balanced parallel edges
        if bal_pedges[0]:
            len_pedges = len(cycle_edges)
            p_ed = bal_pedges[1]  # Number of parallel edges in each set of parallel edges

            for i in range(2, len_pedges + 1):
                nci = Utilities.number_combination(len_pedges, i)
                polynomial += (-1) ** i * nci * (i - 1) * (1 - p) ** (p_ed * i)

            polynomial = 1 - polynomial

        # Inclusion-exclusion formula
        else:
            part = 0
            tmp = 0

            for i in range(2, len(cycle_edges) + 1):
                for subset in itt.combinations(cycle_edges, i):
                    exp = 0
                    for element in subset:
                        exp += cycle_edges[element]
                    tmp += (1 - p) ** exp
                part += (-1) ** i * (i - 1) * tmp
                tmp = 0

            polynomial = 1 - part

            # if polynomial == 2*(-p + 1)**3 - 3*(-p + 1)**2 + 1:  # TODO: Debug test
            #   debug = 1

        # return sympy.Poly(polynomial)  # TODO: Check if works
        return polynomial

    @staticmethod
    def relpoly_multi_treecyc(bal_pedges, tree_edges, cycle_edges, g=None):
        """
        This method combines the methods relpoly_multicycle and relpoly_multitree to get the reliability
        Polynomial of any Tree+Cycles graph (multiedge or not).
        :param bal_pedges: <tuple> (bool, int); <bool> if is parallel, <int> number of parallel edges.
        :param tree_edges: <list> of the edges of the tree part of the graph.
        :param cycle_edges: <list> (list), list of the edges of each cycle of the graph.
        :param g: Graph (only for debugging)
        :return: Reliability polynomial of the graph
        """
        if tree_edges:
            polynomial = GraphRel.relpoly_multitree(bal_pedges, tree_edges)

        else:
            polynomial = 1

        for cycle in cycle_edges:
            polynomial *= GraphRel.relpoly_multicycle(bal_pedges, cycle)

        """
        # For debug purposes
        p = sympy.symbols('p')
        s_ref = GraphRel.relpoly_binary_basic(g)

        if polynomial != s_ref:
            print("Error in graph: ", g.edges, "\n\n Improved Rel m_treecyc:\n", polynomial, polynomial.subs({p: 0.6}), 
            "\n Basic Rel:\n", s_ref, "= ", s_ref.subs({p: 0.6}), "\n\n\n", file=open("deb_output.txt", "a"))
        """

        return polynomial

    @staticmethod
    def relpoly_grid_2xl(g):
        """
        Specialized method to calculates the Rel(G,p) of a 2xl grid graph
        :param g: 2xl grid graph
        :return: Rel(G,p) polynomial
        """
        p = sympy.symbols('p')

        n_edges = len(g.edges)
        n_nodes = len(g.nodes)
        n_cycles = n_edges - n_nodes + 1

        gen_cycle_edges = int((n_edges + n_cycles - 1) / n_cycles)

        prev_p = p
        omegas = 1
        for c in range(0, n_cycles):
            for i in range(0, gen_cycle_edges - 2):
                omegas *= 1 - (1 - p) * (1 - prev_p)
                prev_p = (p * prev_p) / (1 - (1 - p) * (1 - prev_p))

            prev_p = 1 - (1 - p) * (1 - prev_p)

        rel_g = omegas * prev_p

        # return sympy.simplify(rel_g)  # Slower performance
        return rel_g

    @staticmethod
    def relpoly_hypercube(dimension, t_edges, p):
        """
        This mehtod provides the first 3n - 5 coefficients of the reliability polynomial of n-cube networks
        for n >= 4.
        Equation extracted from'On maximal 3-restricted edge connectivity and reliability analysis
        of hypercube networks - Jianping Ou'
        :param dimension: Dimension of the hypercube
        :param t_edges: Graph edges
        :param p: Reliability of the edges
        :return: Reliability polynomial of the first 3n - 5 coefficients
        """
        rel = 0
        for h in range(1, t_edges):
            rel += GraphRel.first_edge_cuts_hypercube(dimension, t_edges, h) \
                   * (1 - p) ** h \
                   * p ** (t_edges - h)

        return 1 - rel

    @staticmethod
    def first_edge_cuts_hypercube(dimension, t_edges, h):
        """
        This method provides the first edge-cuts of any hypercube of dimension >= 4
        Equation extracted from'On maximal 3-restricted edge connectivity and reliability analysis
        of hypercube networks - Jianping Ou'
        :param dimension: Dimension of the hypercube
        :param t_edges: Graph edges
        :param h: Number of edges crossing a cut 'C' (size of the cut)
        :return: Number of edge-cuts with size h
        """

        if h < dimension:
            return 0

        elif (2 * dimension - 3) >= h >= dimension:
            return 2 ** dimension * Utilities.number_combination(t_edges - dimension, h - dimension)

        elif (3 * dimension - 5) >= h >= (2 * dimension - 2):
            return 2 ** dimension * Utilities.number_combination(t_edges - dimension, h - dimension) \
                   + t_edges * Utilities.number_combination(t_edges - 2 * dimension + 1, h - 2 * dimension + 2) \
                   - 2 * Utilities.number_combination(t_edges - 2 * dimension + 1, h - 2 * dimension + 2) \
                   - (Utilities.number_combination(2 ** dimension, 2) - t_edges) \
                   * Utilities.number_combination(t_edges - 2 * dimension, h - 2 * dimension)

        # Error flag: -1
        else:
            return -1

    @staticmethod
    def series_reduction(g, src_node=None):
        """
        This method performs a series reduction. See 'A Linear-Time Algorithm For Computing K-Terminal
        Reliability In Series-Parallel Networks, A. Satyanarayana and R. Kevin Wood
        :param g: Networkx Graph to be reduced
        :param src_node: Set the source node which the algorithm will start with.
        :return: Reducted Graph
        """
        # TODO: Checks if the choosed sub-graph is a proper one.

        graph_nodes = g.nodes()

        if len(graph_nodes) < 2:
            raise ValueError("The graph must contain at least 2 edges")

        if len(graph_nodes) == 3 and len(g.edges()) == 2:
            raise ValueError("The graph cannot be a cycle of 3 nodes")

        if src_node is None:
            src_node = choice(list(graph_nodes))

        # Do a BFS to find 2 adjacent edges from the selected node
        lst_edges = list(nx.bfs_tree(g, source=src_node, depth_limit=2).edges())

        # Get only 2 edges
        sub_lst = lst_edges[0:2]

        sub_graph = g.edge_subgraph(sub_lst).copy()  # Extract a sub-graph using the previous selected 2 edges

        center = nx.center(sub_graph)  # Get the inner node of the 3-path
        c_edge = list(nx.edges(sub_graph, center))  # Get the related edges of the inner node

        # Contract, at the main graph, one of the two inner node related edges
        contracted = nx.contracted_edge(g, c_edge[-1], self_loops=False)

        return contracted, c_edge[-1]

    @staticmethod
    def degree2_reduction(g, src_node=None, omega=1, prohibited_nodes=None):
        """
        This method performs a degree 2 reduction. See 'A Linear-Time Algorithm For Computing K-Terminal
        Reliability In Series-Parallel Networks, A. Satyanarayana and R. Kevin Wood
        :param g: Networkx Graph to be reduced
        :param src_node: Set the source node which the algorithm will start with.
        :param omega: (Only for recursion) Weight of the contracted edge
        :param prohibited_nodes: (Only for recursion) Set of nodes that won't be reduced
        :return: Reducted Graph
        """

        if prohibited_nodes is None:
            prohibited_nodes = set()

        graph_nodes = set(g.nodes())
        graph_nodes -= prohibited_nodes

        # Restrictions
        if len(graph_nodes) < 3:
            raise ValueError("The graph must contain at least 3 nodes")

        if len(g.edges) < 2:
            raise ValueError("The graph must contain at least 2 edges")

        if src_node is None:
            src_node = choice(tuple(graph_nodes))

        # Do a BFS to find 2 adjacent edges from the selected node
        lst_edges = list(nx.bfs_tree(g, source=src_node, depth_limit=2).edges())

        sub_lst = list()
        for edge in lst_edges:
            if g.number_of_edges(edge[0], edge[1]) <= 1:
                sub_lst.append((edge[0], edge[1], 0))

        # If we cannot find at least 2 non-parallel edges with the given source node, then choose another
        if len(sub_lst) < 2:
            prohibited_nodes.add(src_node)
            return GraphRel.degree2_reduction(g, None, omega, prohibited_nodes)

        # Get only 2 edges
        sub_lst = sub_lst[0:2]

        # In this case the graph is not reducible
        if len(sub_lst) < 2:
            return g, 1

        """
        Contract, at the main graph, the second edge of the sub_list (the previous selected edge)
        and save the result of the contraction to a new variable ('contracted')
        """
        contracted = nx.contracted_edge(g, (sub_lst[-1][0], sub_lst[-1][1]), self_loops=False)

        # Get the probabilities of the contracted edges
        edge_p1 = g[sub_lst[0][0]][sub_lst[0][1]][sub_lst[0][2]]['prob']
        edge_p2 = g[sub_lst[-1][0]][sub_lst[-1][1]][sub_lst[-1][2]]['prob']

        # Add or modify the weights (p and omega) to the contracted edge
        contracted[sub_lst[0][0]][sub_lst[0][1]][sub_lst[0][2]]['prob'] = (edge_p1 * edge_p2) / (
                1 - (1 - edge_p1) * (1 - edge_p2))

        omega *= (1 - (1 - edge_p1) * (1 - edge_p2))

        return contracted, omega

    @staticmethod
    def series_parallel_alg(g):
        # TODO: Not finished algorithm
        """
        This method performs a series parallel reduction. See 'A Linear-Time Algorithm For Computing K-Terminal
        Reliability In Series-Parallel Networks, A. Satyanarayana and R. Kevin Wood
        :param g: Networkx Graph to be reduced
        :return: The reduced graph
        """

        # Restrictions
        if not nx.is_connected():
            raise ValueError("The given graph is not connected")

        if len(g.nodes) < 2:
            raise ValueError("The given graph must have at least 2 nodes")

        if len(g.edges) < 2:
            raise ValueError("The given graph must have at least 2 edges")

        test_edge = choice(list(g.edges))
        if 'prob' in g[test_edge[0]][test_edge[1]]:
            raise ValueError("The graph edges needs to have associated probabilities 'prob' ")

        # Perfom all degree-2 reductions
        g_red, omega = GraphRel.degree2_reduction(g)

        if g_red is None:
            g_red = g
            omega = 1

        else:
            while g_red is not None:
                prev_g = g_red
                prev_o = omega
                t_red, omega = GraphRel.degree2_reduction(g_red)
            g_red = prev_g
            omega = prev_o

        # Construct list, T <- {v|v included V and deg(v)>2} making all such v "onlist" and marking the others "offlist"
        on_list = list()
        off_list = list()
        for vertex in list(g.nodes):
            if g.degree(vertex) > 2:
                on_list.append(vertex)
            else:
                off_list.append(vertex)

        while on_list != list() and len(g.edges) > 2:
            v = on_list.pop()
            i = 1  # Index of the next chain out of v to be searched

            while i < 4 or v is None or g.degree(v) == 2:
                # TODO: Search the ith chain out of v
                i += 1
                # TODO: Find a polygon (u,w)

        if len(g.edges) == 2:
            print("R(G_k) is ")  # TODO: Finish print

        else:
            print("G is not series-parallel")

        return -1

    @staticmethod
    def fair_cake_algorithm(n_nodes, chords):
        """
        Algorithm that constructs a Fair Cake Graph (FCG) with the given nodes and chords.
        :param n_nodes: Number of nodes
        :param chords: Number of chords
        :return: FCG
        """

        cycle = nx.cycle_graph(n_nodes)
        chorded_cycle = copy.deepcopy(cycle)
        nodes = list(cycle.nodes)
        n_edges = len(cycle.edges)
        # chord_list = list() # Debug

        #  Not enough edges to form a cycle with all the nodes
        if chords < 0:
            raise ValueError("The number of edges must be at least equal to the number of nodes\n")

        elif n_edges > (n_nodes + n_nodes / 2):
            raise ValueError("The number of edges cannot be greater than n_nodes + n_nodes/2")

        # Just enough edges to form a cycle with all the nodes
        elif chords == 0:
            return cycle

        separation = n_nodes / (chords * 2)

        mid = int(len(nodes) / 2)
        position = abs(separation - 1)  # -1 because we count position 0
        while position < mid:
            chorded_cycle.add_edge(nodes[int(position)], nodes[int(position + mid)])
            # chord_list.append((nodes[int(position)], nodes[int(position + mid)]))  # Debug: List of chords
            position = position + separation

        return chorded_cycle

    @staticmethod
    def fair_cake_defiance(ham_graph, check_polynomials=True):
        """
        Method that checks which hamiltonian graph with 3 chords is better than the FC construction
        Input: hamiltonian graph with 3 chords
        :return: File with the results
        """
        p = sympy.symbols("p")

        print("Starting...")

        ham_cycle_nodes = GraphTools.hamilton_cycle(ham_graph)

        n = len(ham_cycle_nodes)
        chords = set()
        for i in range(0, len(ham_cycle_nodes)):
            if ham_graph.degree(ham_cycle_nodes[i]) == 3:
                for node in ham_graph.neighbors(ham_cycle_nodes[i]):
                    if node != ham_cycle_nodes[i - 1] and node != ham_cycle_nodes[(i + 1) % n]:
                        if (node, ham_cycle_nodes[i]) not in chords:
                            chords.add((ham_cycle_nodes[i], node))
                            break

        if os.path.isfile(os.getcwd() + "/Data/Results/" + str(len(ham_graph)) + "n_" +
                          str(len(ham_graph.edges) - len(ham_graph)) + "ch_better_than_FC.txt"):
            os.remove(os.getcwd() + "/Data/Results/" + str(len(ham_graph)) + "n_" +
                      str(len(ham_graph.edges) - len(ham_graph)) + "ch_better_than_FC.txt")

        if check_polynomials:
            ref_poly = GraphRel.relpoly_binary_improved(ham_graph, 0)
            ref_avg_p = sympy.integrate(ref_poly.as_expr(), (p, 0, 1))

            message = ("Original graph ;"
                       "Hamiltonian cycle ;"
                       "Polynomial ;"
                       "Avg_polynomial ;")

            data = (str(list(ham_graph.edges)) + "; " +
                    str(ham_cycle_nodes) + "; " +
                    str(ref_poly) + "; " +
                    str(ref_avg_p))

        else:
            ref_sp = GraphTools.spanning_trees_count(ham_graph)
            ref_ec = nx.edge_connectivity(ham_graph)
            ref_mk = len(GraphTools.minimum_k_edges_cutsets(ham_graph, 2))

            message = ("Original graph ;"
                       "Hamiltonian cycle ;"
                       "Spanning Trees; "
                       "Edge connectivity; "
                       "Min. k=2 edge-cut ")

            data = (str(list(ham_graph.edges)) + "; " +
                    str(ham_cycle_nodes) + "; " +
                    str(ref_sp) + ";" +
                    str(ref_ec) + ";" +
                    str(ref_mk))

        print(message, file=open(os.getcwd() + "/Data/Results/" + str(
            len(ham_graph)) + "n_" + str(len(ham_graph.edges) - len(ham_graph)) + "ch_better_than_FC.txt", "a"))

        print(data, file=open(os.getcwd() + "/Data/Results/" + str(
            len(ham_graph)) + "n_" + str(len(ham_graph.edges) - len(ham_graph)) + "ch_better_than_FC.txt", "a"))

        if check_polynomials:
            message = ("New graph ;"
                       "Hamiltonian cycle ;"
                       "Optimal if ;"
                       "Polynomial ;"
                       "Avg_polynomial ;")

        else:
            message = ("New graph ;"
                       "Hamiltonian cycle ;"
                       "Optimal if ;"
                       "Spanning Trees; "
                       "Edge connectivity; "
                       "Min. k=2 edge-cut ")

        print(message, file=open(os.getcwd() + "/Data/Results/" + str(
            len(ham_graph)) + "n_" + str(len(ham_graph.edges) - len(ham_graph)) + "ch_better_than_FC.txt", "a"))

        for chord in chords:
            for x in range(0, len(ham_cycle_nodes)):
                for y in range(x + 2, len(ham_cycle_nodes) - 1 if x == 0 else len(ham_cycle_nodes)):
                    if (x, y) != chord:
                        tmp_graph = copy.deepcopy(ham_graph)
                        tmp_graph.remove_edge(chord[0], chord[1])

                        if not ham_graph.has_edge(x, y):
                            tmp_graph.add_edge(x, y)

                            if check_polynomials:
                                poly = GraphRel.relpoly_binary_improved(tmp_graph, 0)
                                avg_p = sympy.integrate(poly.as_expr(), (p, 0, 1))

                                if avg_p > ref_avg_p:
                                    data = (str(list(tmp_graph.edges)) + "; " +
                                            str(GraphTools.hamilton_cycle(tmp_graph)) + "; " +
                                            "-" + str(chord) + "+" + str((x, y)) + "; " +
                                            str(poly) + "; " +
                                            str(avg_p))

                                    print(data, file=open(os.getcwd() + "/Data/Results/" + str(
                                        len(ham_graph)) + "n_" + str(
                                        len(ham_graph.edges) - len(ham_graph)) + "ch_better_than_FC.txt", "a"))

                            else:
                                sp = GraphTools.spanning_trees_count(tmp_graph)
                                ec = nx.edge_connectivity(tmp_graph)
                                mk = len(GraphTools.minimum_k_edges_cutsets(tmp_graph, 2))

                                if sp > ref_sp \
                                        or sp == ref_sp and ec > ref_ec \
                                        or sp == ref_sp and ec == ref_ec and mk < ref_mk:
                                    data = (str(list(tmp_graph.edges)) + "; " +
                                            str(GraphTools.hamilton_cycle(tmp_graph)) + "; " +
                                            "-" + str(chord) + "+" + str((x, y)) + "; " +
                                            str(sp) + ";" +
                                            str(ec) + ";" +
                                            str(mk))
                                    print(data, file=open(os.getcwd() + "/Data/Results/" + str(
                                        len(ham_graph)) + "n_" + str(len(ham_graph.edges) - len(ham_graph))
                                                          + "ch_better_than_FC.txt", "a"))

        print("Done")

    @staticmethod
    def opt_6k_c_path(k, plus, ch):
        """
        This method calculates the optimal c-path vector for hamiltonian cycle graphs which nodes are multiple
        of 6 with an addition (plus) of 2, 3 or 4 and a maximum number of chords of 3.
        :param k: variable that defines the multiple of 6
        :param plus: the addition to the number of nodes
        :param ch: number of chords
        :return: the optimum vector of c-path lengths
        """

        # Restrictions
        if ch > 3:
            raise NotImplementedError

        if plus == 2 and k == 1:
            raise ValueError('K must be greater than 1 for graphs n=6k+2')

        if plus == 1 or 5:
            raise ValueError('The addition must be 2, 3 or 4')

        n_nodes = 6 * k

        # No chords
        if ch == 0:
            return [n_nodes]

        c_lenght = ch * 2  # Number of paths

        # Calculate the number of k and k+1 paths
        nk_1 = modf((6 * k + plus) / c_lenght)[0] * c_lenght
        nk = c_lenght - nk_1

        # Construct the vector
        c_path = []
        while len(c_path) < c_lenght:
            if nk_1 > 0:
                c_path.append(int((n_nodes / c_lenght) + 1))
                nk_1 -= 1

            if nk > 0:
                c_path.append(int(n_nodes / c_lenght))
                nk -= 1

        return c_path
