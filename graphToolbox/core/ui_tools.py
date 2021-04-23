__author__ = 'Pol Llagostera Blasco'

# Project classes
from graphToolbox.core.graphtbox import *


def create_data_dirs():
    path = os.getcwd()
    if not os.path.exists(path + "/data"):
        os.makedirs(path + "/data")
    if not os.path.exists(path + "/data/databases"):
        os.makedirs(path + "/data/databases")
    if not os.path.exists(path + "/data/graph6"):
        os.makedirs(path + "/data/graph6")
    if not os.path.exists(path + "/data/plain_results"):
        os.makedirs(path + "/data/plain_results")
    if not os.path.exists(path + "/data/test_results"):
        os.makedirs(path + "/data/test_results")
    if not os.path.exists(path + "/data/tmp"):
        os.makedirs(path + "/data/tmp")


def file_len(file_path):
    """
    Gives the number of lines of the file
    :param file_path: path to the file
    :return: number of files
    """
    with open(file_path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def input_g6_file(message):
    """
    Simple function to ask the user to input a .g6 file name.
    :param message: Message that will be shown to the user.
    :return: List of the read networkx graphs and the given file name
    """
    path = os.getcwd() + "/data/graph6/"

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


def input_file(message, path):
    """
        Simple function to check if the input file exist.
        :param path: path to the file
        :param message: Message that will be shown to the user.
        :return: the given file name
        """
    while True:
        file_name = input(message + "\n")
        if os.path.isfile(path + file_name + ".g6"):
            return file_name

        else:
            print("The file ", file_name, " doesn't exist, please make sure that is in the folder ", path)


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

class Switcher(object):
    """
    This class defines the methods used by each option of the ui.
    """

    def option_gen_g6_graphs(self):
        print("This option will generate graphs within the range of ['n_min', 'n_max] nodes "
              "and e=n/2 (N) or e=n(n − 1)/2 (Y) edges")
        n_min = int(input_number("Input minimum nodes:\n"))
        n_max = int(input_number("Input maximum nodes:\n"))

        complete = ask_yes_no("Generate edges until the complete graph?")

        GraphTools.gen_g6_files(n_min, n_max, complete)

    def option_gen_g6_hamiltonian_graphs(self):
        # TODO: Change chords by edges
        print("This option will generate hamiltonian graphs within the range of ['n_min', 'n_max] nodes "
              "and 'ch' chords")

        n_nodes = int(input_number("Input number of nodes:\n"))
        chords = int(input_number("Input number of chords:\n"))

        GraphTools.gen_all_hamiltonian(n_nodes, chords, True)

    def option_gen_fair_cake(self):
        print("This option will give all the fair cake graph constructions with the given 'n' nodes ")
        n_nodes = int(input_number("Input number of nodes:\n"))
        analyze = ask_yes_no("Analyze the created graphs? (Reliability, Hamiltonian path, etc.)")

        path = os.getcwd() + "/data"

        if not analyze:
            if os.path.isfile(path + "/graph6/" + str(n_nodes) + "n_FairCake.g6"):
                os.remove(path + "/graph6/" + str(n_nodes) + "n_FairCake.g6")

        g_list = list()
        for i in range(1, int(n_nodes / 2) + 1):
            print("Constructing Fair Cake: " + str(n_nodes) + "n, " + str(i + n_nodes) + "e")
            fc = GraphRel.fair_cake_algorithm(n_nodes, i)
            g_list.append(fc)

            if not analyze:
                nx.write_graph6(fc, path + "/graph6/" + str(n_nodes) + "n_FairCake_tmp.g6", header=False)

                with open(path + "/graph6/" + str(n_nodes) + "n_FairCake.g6", 'a') as outfile:
                    with open(path + "/graph6/" + str(n_nodes) + "n_FairCake_tmp.g6") as infile:
                        outfile.write(infile.read())
                os.remove(path + "/graph6/" + str(n_nodes) + "n_FairCake_tmp.g6")

        if analyze:
            GraphTools.analyze_graphs(g_list, path, str(n_nodes) + "n_FairCake", False)

        else:
            print("Graphs stored in g6 format into the file " + path + "/graph6/" + str(n_nodes) + "n_FairCake.g6")

    def option_analyze_a_g6_file(self):
        print("This option will analyze graphs inside a .g6 file.")

        g_list, file_name = input_g6_file("Enter the name of the .g6 file.")
        fast = ask_yes_no("Do you want to execute a fast but with less information analysis?")
        cros = False if fast else ask_yes_no("Check if the reliability polynomials cross in some point?")
        GraphTools.analyze_g6_graphs(file_name, cros, g_list, fast)

    def option_analyze_multiple_g6(self):
        print("This option will analyze the graphs in .g6 format within the range of ['n_min', 'n_max] nodes, "
              "and e=n/2 (N) or e=n(n − 1)/2 (Y) edges")
        n_min = int(input_number("Input minimum nodes:\n"))
        n_max = int(input_number("Input maximum nodes:\n"))

        complete = ask_yes_no("Analyze the graphs with edges at the range ('n' to complete graph)?")
        cros = ask_yes_no("Check if the polynomials cross in some point?")

        GraphTools.analyze_g6_files(n_min, n_max, complete, cros)

    def option_get_multiple_optimal(self):
        path = os.getcwd() + "/data"
        g_list = list()

        print("From previously stored graphs in .g6 format, this option will filter the most reliable ones."
              "A range of ['n_min', 'n_max] nodes, and e=n/2 (N) or e=n(n − 1)/2 (Y) edges, must be given in "
              "order to choose the files.")
        n_min = int(input_number("Input minimum nodes:\n"))
        n_max = int(input_number("Input maximum nodes:\n"))

        complete = ask_yes_no("Analyze the graphs with edges at the range ('n' to complete graph)?")

        for nodes in range(n_min, n_max + 1):
            # Complete graph
            if complete:
                edges = int(nodes * (nodes - 1) / 2)

            # Gives -> nodes/2 <- chords
            else:
                edges = int(nodes / 2 + nodes)

            for i in range(nodes, edges + 1):
                file_name = "Graph_n" + str(nodes) + "_e" + str(i)
                print("\nAnalyzing graph:", file_name)
                if os.path.isfile(path + "/graph6/" + file_name + ".g6"):
                    aux_lst = nx.read_graph6(path + "/graph6/" + file_name + ".g6")
                    if type(aux_lst) is not list:
                        aux_lst = [aux_lst]

                    g_list += GraphTools.filter_uniformly_most_reliable_graph(aux_lst)
                else:
                    print("\n File ", file_name, ".g6 not found.")

        GraphTools.analyze_graphs(g_list, path, "Multiple_opt_graphs", False)

        print("\nDone")

    def option_get_reliability_polynomial_optimal_graphs(self):
        print("This option will give the most reliable graphs constructions (hamiltonian and non-hamiltonians)"
              " of 'n' nodes and e=n/2 (N) or e=n(n − 1)/2 (Y) edges. The results will be written in"
              "different files.")

        n_nodes = int(input_number("Input number of nodes:\n"))
        complete = ask_yes_no("Analyze the graphs with edges at the range ('n' to complete graph)?")
        GraphTools.reliability_polynomial_optimal_graphs(n_nodes, complete)

    def option_graphs_comparison(self):
        print("This option will compare a graph with a bunch of graphs. All files must be written in .g6 format.")
        graph1 = input_g6_file("Input the name of the .g6 file containing the graph to compare")
        if len(graph1) == 0 or len(graph1) > 1:
            print("The file has to contain only one graph")
            return None
        graph_bunch = input_g6_file("Input the name of the .g6 file containing the second graph")

        file_name = input("Please input the name of the results file\n")

        GraphTools.compare_graphs(file_name, graph1[0], graph_bunch)

    def option_compare_best_reliabilities(self):
        print("This option will filter the best Reliability Polynomial of General graphs and the best of the"
              "Hamiltonian ones. Then will analyze and compare them. \nNotice that the graphs must be written in the"
              "same .g6 file and they must have the same nodes and the same edges\n")

        g_list, file_name = input_g6_file("Input the .g6 file containing the graphs.")
        filtered_ham, filtered_others = GraphTools.filter_hamiltonian_general_graphs(g_list)

        if len(filtered_ham) > 1 or len(filtered_others) > 1:
            print("WARNING: The filtered graphs gives more than one optimal")

        if len(filtered_ham) == 1 and len(filtered_others) == 0:
            print("The given graphs are only Hamiltonians")

        if len(filtered_ham) == 0 and len(filtered_others) == 1:
            print("Any of the given graphs are Hamiltonians")

        else:
            GraphTools.compare_graphs("Optimals_" + file_name, filtered_ham[0], filtered_others[0])

    def option_range_compare_best_reliabilities(self):
        print("This option will filter the best Reliability Polynomial of General graphs and the best of "
              "Hamiltonian ones. Then will analyze and compare them. \nThis filter will be applied to the graphs "
              "(written in .g6 format) within a range of ['n_min', 'n_max] nodes, and e=n/2 (N) or e=n(n − 1)/2 (Y) "
              "edges")

        path = os.getcwd() + "/data/graph6/"

        n_min = int(input_number("Input minimum nodes:\n"))
        n_max = int(input_number("Input maximum nodes:\n"))

        complete = ask_yes_no("Analyze the graphs with edges at the range ('n' to complete graph)?")

        for nodes in range(n_min, n_max + 1):
            # Complete graph
            if complete:
                edges = int(nodes * (nodes - 1) / 2)

            # Gives -> nodes/2 <- chords
            else:
                edges = int(nodes / 2 + nodes)

            for i in range(nodes, edges + 1):
                file_name = "Graph_n" + str(nodes) + "_e" + str(i)
                print("\nAnalyzing graph:", file_name)

                if os.path.isfile(path + file_name + ".g6"):
                    g_list = nx.read_graph6(path + file_name + ".g6")

                    # If the read graphs is only one, wrap it with a list
                    if type(g_list) is not list:
                        g_list = [g_list]

                    filtered_ham, filtered_others = GraphTools.filter_hamiltonian_general_graphs(g_list)

                    if len(filtered_ham) > 1 or len(filtered_others) > 1:
                        print("WARNING: The filtered graphs gives more than one optimal")

                    if len(filtered_ham) == 1 and len(filtered_others) == 0:
                        print("The given graphs are only Hamiltonians")

                    if len(filtered_ham) == 0 and len(filtered_others) == 1:
                        print("None of the given graphs are Hamiltonians")

                    else:
                        GraphTools.compare_graphs("Optimals_" + file_name, filtered_ham[0], filtered_others[0])

                else:
                    print("\n File ", file_name, ".g6 not found.")

    def option_compare_coefficients(self):
        print("This option will check if all the coefficients of the first binomial are greater "
              "than the ones of the second binomial")
        graph1 = input_g6_file("Input the name of the .g6 file containing the graph to compare")

        if len(graph1[0]) == 0 or len(graph1[0]) > 1:
            print("The file has to contain only one graph")
            return None

        graph_bunch = input_g6_file("Input the name of the .g6 file containing the second graph")

        filtered_ham, filtered_others = GraphTools.filter_hamiltonian_general_graphs(graph_bunch[0], False)

        GraphTools.compare_graphs("Coef" + graph1[1], graph1[0][0], filtered_ham, False, True)

    def option_gen_all_3ch_hamiltonian_opt(self):
        print("This option will analyze hamiltonian graphs with a maximum of 3 chords (e=n+3). At least one of the chords "
              "will be diametrical and the other 2 will cross it. The .g6 file containing the diametral graphs must be created"
              "previously with name 'Diametral_n<number of nodes>_ch<number of chords>.g6")

        n_min = int(input_number("Input minimum nodes:\n"))
        n_max = int(input_number("Input maximum nodes:\n"))

        for n in range(n_min, n_max + 1):
            # hamiltonians = GraphTools.gen_all_3ch_hamiltonian_opt(n)
            hamiltonians = nx.read_graph6(os.getcwd() + "/data/graph6/Diametral_n" + str(n) + "_ch3.g6")
            print("\nAnalyzing ", len(hamiltonians), "hamiltonian graphs with: ", n, "nodes")
            dfs = None
            for ham in hamiltonians:
                df = GraphTools.data_analysis(ham, True)
                if dfs is None:
                    dfs = df
                else:
                    dfs = dfs.append(df)

            GraphTools.data_print(dfs, FormatType.SQL, os.getcwd() + "/data/databases/" + "Graphs_DB")

            print("Analyzed")

    def option_testing(self):
        print("This option will the analyze graphs inside a .g6 file and print the results in the desired format.")

        g_list, file_name = input_g6_file("Enter the name of the .g6 file.")
        fast = ask_yes_no("Do you want to execute a fast but with less information analysis?")

        df = GraphTools.data_analysis(g_list[0], True, fast)

        write_format = input_data_format("Select the saving format for the resulting data:\n"
                                                 "1) CSV\n"
                                                 "2) Excel\n"
                                                 "3) HTML\n"
                                                 "4) JSON\n"
                                                 "5) SQL")

        GraphTools.data_print(df, write_format, os.getcwd() + "/data/databases/" + "Graphs_DB")
        """
        GraphTools.data_print(df, FormatType.CSV, os.getcwd() + "/data/databases/" + "Graphs_DB")
        GraphTools.data_print(df, FormatType.JSON, os.getcwd() + "/data/databases/" + "Graphs_DB")
        GraphTools.data_print(df, FormatType.HTML, os.getcwd() + "/data/databases/" + "Graphs_DB")
        #GraphTools.data_print(df, FormatType.Excel, os.getcwd() + "/data/databases/" + "DB_file_name")
        """

    def option_g6_file_data_analysis2db(self):
        print("This option will analyze the graphs inside a .g6 file and save the results into a SQLite data base "
              "called 'Graphs_DB'")

        path = os.getcwd() + "/data/graph6/"
        file_name = input_file("Enter the name of the .g6 file.", path)
        fast = ask_yes_no("Do you want to execute a fast but with less information analysis?")

        n_lines = file_len(path + file_name + ".g6")

        GraphTools.g6_file_data_analysis2db(file_name, fast, n_lines)


    def option_g6_files_data_analysis2db(self):
        print("This option will analyze the graphs in .g6 format within a range of ['n_min', 'n_max] nodes, "
              "and e=n/2 (N) or e=n(n − 1)/2 (Y) edges and save the results into a SQLite data base called 'Graphs_DB'")
        n_min = int(input_number("Input minimum nodes:\n"))
        n_max = int(input_number("Input maximum nodes:\n"))

        # complete = Utilities.ask_yes_no("Analyze the graphs with edges at the range (n to complete graph)?")
        chords = int(input_number("Input maximum number of chords, or 0 for complete graph:\n"))

        time_start = time.process_time()
        GraphTools.g6_files_data_analysis2db(n_min, n_max, chords)
        time_elapsed = (time.process_time() - time_start)
        print("\nTime: ", time_elapsed)
