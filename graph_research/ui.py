from graph_research.core.graphtbox import *


class Switcher(object):

    def option_gen_g6_graphs(self):
        print("This option will generate graphs within the range of 'n' nodes (minimum-maximum). "
              "and depending of the selected option; 'e' edges (n to n/2) or (n to complete graph)")
        n_min = int(Utilities.input_number("Input minimum nodes:\n"))
        n_max = int(Utilities.input_number("Input maximum nodes:\n"))

        complete = Utilities.ask_yes_no("Generate edges until the complete graph?")

        GraphTools.gen_g6_files(n_min, n_max, complete)

    def option_gen_g6_hamiltonian_graphs(self):
        print("This option will generate hamiltonian graphs within the range of 'n' nodes (minimum-maximum) "
              "and 'ch' chords")

        n_nodes = int(Utilities.input_number("Input number of nodes:\n"))
        chords = int(Utilities.input_number("Input number of chords:\n"))

        g_list = GraphTools.gen_all_hamiltonian(n_nodes, chords)
        GraphTools.gen_g6_file(g_list, "Hamilton_n" + str(n_nodes)+"_ch" + str(chords))

    def option_gen_fair_cake(self):
        print("This option will give all the fair cake constructions of the given 'n' nodes graph")
        n_nodes = int(Utilities.input_number("Input number of nodes:\n"))
        analyze = Utilities.ask_yes_no("Analyze the created graphs?")

        path = os.getcwd() + "/Data"

        if not analyze:
            if os.path.isfile(path + "/Results/" + str(n_nodes) + "n_FairCake" + "_results.txt"):
                os.remove(path + "/Results/" + str(n_nodes) + "n_FairCake" + "_results.txt")
            print("Graph; Hamiltonian cycle; Graph Edges",
                  file=open(path + "/Results/" + str(n_nodes) + "n_FairCake" + "_results.txt", "a"))
        g_list = list()
        for i in range(1, int(n_nodes/2) + 1):
            print("Constructing Fair Cake: " + str(n_nodes) + "n, " + str(i + n_nodes) + "e")
            fc = GraphRel.fair_cake_algorithm(n_nodes, i)
            g_list.append(fc)

            if not analyze:
                print("\n" + "Graph_n" + str(len(fc.nodes)) + "_e" + str(len(fc.edges)) + ";" +
                      str(GraphTools.hamilton_cycle(fc)) + ";" +
                      str(sorted(fc.edges())),
                      file=open(path + "/Results/" + str(n_nodes) + "n_FairCake" + "_results.txt", "a"))

        if analyze:
            GraphTools.analyze_graphs(g_list, path, str(n_nodes) + "n_FairCake", False)

    def option_analyze_a_g6_file(self):
        print("This option will the analyze graphs inside a .g6 file.")

        g_list, file_name = Utilities.input_g6_file("Enter the name of the .g6 file.")
        fast = Utilities.ask_yes_no("Do you want to execute a fast but with less information analysis?")
        cros = False if fast else Utilities.ask_yes_no("Check if the polynomials cross in some point?")
        GraphTools.analyze_g6_graphs(file_name, cros, g_list, fast)

    def option_analyze_multiple_g6(self):
        print("This option will analyze the graphs in .g6 format within the range of 'n' nodes (minimum-maximum), "
              "and depending of the selected option; 'e' edges (n to n/2) or (n to complete graph)")
        n_min = int(Utilities.input_number("Input minimum nodes:\n"))
        n_max = int(Utilities.input_number("Input maximum nodes:\n"))

        complete = Utilities.ask_yes_no("Analyze the graphs with edges at the range (n to complete graph)?")
        cros = Utilities.ask_yes_no("Check if the polynomials cross in some point?")

        GraphTools.analyze_g6_files(n_min, n_max, complete, cros)

    def option_get_multiple_optimal(self):
        path = os.getcwd() + "/Data"
        g_list = list()

        print("This option will filter the most reliable graphs in .g6 format within the range of "
              "'n' nodes (minimum-maximum), and depending of the selected option; "
              "'e' edges (n to n/2) or (n to complete graph)")
        n_min = int(Utilities.input_number("Input minimum nodes:\n"))
        n_max = int(Utilities.input_number("Input maximum nodes:\n"))

        complete = Utilities.ask_yes_no("Analyze the graphs with edges at the range (n to complete graph)?")

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
                if os.path.isfile(path + "/Graph6/" + file_name + ".g6"):
                    aux_lst = nx.read_graph6(path + "/Graph6/" + file_name + ".g6")
                    if type(aux_lst) is not list:
                        aux_lst = [aux_lst]

                    g_list += GraphTools.filter_uniformly_most_reliable_graph(aux_lst)
                else:
                    print("\n File ", file_name, ".g6 not found.")

        GraphTools.analyze_graphs(g_list, path, "Multiple_opt_graphs")

        print("\nDone")

    def option_get_reliability_polynomial_optimal_graphs(self):
        print("This option will give the most reliable hamiltonian and general graphs constructions "
              "(wrote in different files) of 'n' nodes to 'e' edges (n to n/2) or (n to complete graph)")

        n_nodes = int(Utilities.input_number("Input number of nodes:\n"))
        complete = Utilities.ask_yes_no("Analyze the graphs with edges at the range (n to complete graph)?")
        GraphTools.reliability_polynomial_optimal_graphs(n_nodes, complete)


    def option_graphs_comparison(self):
        print("This option will compare one graph with a bunch of graphs. All files must be wrote in .g6 format")
        graph1 = Utilities.input_g6_file("Input the name of the .g6 file containing the graph to compare")
        if len(graph1) == 0 or len(graph1) > 1:
            print("The file has to contain only one graph")
            return None
        graph_bunch = Utilities.input_g6_file("Input the name of the .g6 file containing the second graph")

        file_name = input("Please input the name of the results file\n")

        GraphTools.compare_graphs(file_name, graph1[0], graph_bunch)

    def option_compare_best_reliabilities(self):
        print("This option will filter the best Reliability Polynomial of General graphs and the best of "
              "Hamiltonian ones. Then will analyze and compare them. Notice that the graphs must be written in the"
              "same .g6 file and they must have the same nodes and the same edges")

        g_list, file_name = Utilities.input_g6_file("Input the .g6 file containing the graphs.")
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
              "Hamiltonian ones. Then will analyze and compare them. This filter will be applied to the graphs "
              "(written in .g6 format) within the range of 'n' nodes (minimum-maximum), "
              "and depending of the selected option; 'e' edges (n to n/2) or (n to complete graph)")

        path = os.getcwd() + "/Data/Graph6/"

        n_min = int(Utilities.input_number("Input minimum nodes:\n"))
        n_max = int(Utilities.input_number("Input maximum nodes:\n"))

        complete = Utilities.ask_yes_no("Analyze the graphs with edges at the range (n to complete graph)?")

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
        print("This option will check if all the coefficients of the first binomial polynomial are greater "
              "than the ones of the second binomial polynomial")
        graph1 = Utilities.input_g6_file("Input the name of the .g6 file containing the graph to compare")

        if len(graph1[0]) == 0 or len(graph1[0]) > 1:
            print("The file has to contain only one graph")
            return None

        graph_bunch = Utilities.input_g6_file("Input the name of the .g6 file containing the second graph")

        filtered_ham, filtered_others = GraphTools.filter_hamiltonian_general_graphs(graph_bunch[0], False)

        GraphTools.compare_graphs("Coef" + graph1[1], graph1[0][0], filtered_ham, False, True)

    def option_gen_all_3ch_hamiltonian_opt(self):
        print("This option will analyze hamiltonian graphs with a maximum of 3 chords. At least one of the chords will"
              "be diametrical and the other 2 will cross it.")

        n_min = int(Utilities.input_number("Input minimum nodes:\n"))
        n_max = int(Utilities.input_number("Input maximum nodes:\n"))

        for n in range(n_min, n_max+1):
            #hamiltonians = GraphTools.gen_all_3ch_hamiltonian_opt(n)
            hamiltonians = nx.read_graph6(os.getcwd() + "/Data/Graph6/Diametral_n" + str(n) + "_ch3.g6")
            print("\nAnalyzing ", len(hamiltonians),"hamiltonian graphs with: ", n, "nodes")
            dfs = None
            for ham in hamiltonians:
                df = GraphTools.data_analysis(ham, True)
                if dfs is None:
                    dfs = df
                else:
                    dfs = dfs.append(df)

            GraphTools.data_print(dfs, FormatType.SQL, os.getcwd() + "/Data/DDBB/" + "Graphs_DB")

            print("Analyzed")


    def option_testing(self):
        print("This option will the analyze graphs inside a .g6 file and print the results in the desired format.")

        g_list, file_name = Utilities.input_g6_file("Enter the name of the .g6 file.")
        fast = Utilities.ask_yes_no("Do you want to execute a fast but with less information analysis?")

        df = GraphTools.data_analysis(g_list[0], True, fast)


        write_format = Utilities.input_data_format("Select the saving format for the resulting data:\n"
                                                   "1) CSV\n"
                                                   "2) Excel\n"
                                                   "3) HTML\n"
                                                   "4) JSON\n"
                                                   "5) SQL")

        GraphTools.data_print(df, write_format, os.getcwd() + "/Data/DDBB/" + "Graphs_DB")
        """
        GraphTools.data_print(df, FormatType.CSV, os.getcwd() + "/Data/DDBB/" + "Graphs_DB")
        GraphTools.data_print(df, FormatType.JSON, os.getcwd() + "/Data/DDBB/" + "Graphs_DB")
        GraphTools.data_print(df, FormatType.HTML, os.getcwd() + "/Data/DDBB/" + "Graphs_DB")
        #GraphTools.data_print(df, FormatType.Excel, os.getcwd() + "/Data/DDBB/" + "DB_file_name")
        """

    def option_g6_files_data_analysis(self):
        print("This option will analyze the graphs in .g6 format within the range of 'n' nodes (minimum-maximum), "
              "and depending of the selected option; 'e' edges (n to n/2) or (n to complete graph) and save the results"
              "into a SQLite data base called 'Graphs_DB'")
        n_min = int(Utilities.input_number("Input minimum nodes:\n"))
        n_max = int(Utilities.input_number("Input maximum nodes:\n"))

        #complete = Utilities.ask_yes_no("Analyze the graphs with edges at the range (n to complete graph)?")
        chords = int(Utilities.input_number("Input maximum number of chords, or 0 for complete graph:\n"))

        time_start = time.process_time()
        GraphTools.g6_files_data_analysis(n_min, n_max, chords)
        time_elapsed = (time.process_time() - time_start)
        print("\nTime: ", time_elapsed)



# ----------------------------------------------------USER INTERFACE----------------------------------------------------

def run():
    sw = Switcher()

    while True:
        option = Utilities.input_number \
            ("Select option:\n"
             "1) Generate graphs in .g6 format.\n"
             "2) Generate Hamiltonian graphs in .g6 format.\n"
             "3) Generate Fair Cake graphs\n"
             
             "4) Analyze a .g6 file.\n"
             "5) Analyze a range of .g6 files.\n"
             "6) Get the most reliable graphs from a range of .g6 files\n"
             "7) Get the most reliable Hamiltonian and General graph constructions\n"
             
             "8) Compare two graphs.\n"
             "9) Compare the best Reliability Polynomial between General and Hamiltonian graphs.\n"
             "10) Compare the best Reliability Polynomial between General and Hamiltonian graphs on a range of graphs.\n"
             "11) Compare the coefficients of one graph with the ones of a bunch of graphs.\n"
             
             "12) Analyze 3 chords hamiltonian graphs.\n"
             
             "13) Testing: Analysis and print data.\n"
             "0) Exit.\n")

        if option == 1:
            sw.option_gen_g6_graphs()

        elif option == 2:
            sw.option_gen_g6_hamiltonian_graphs()

        elif option == 3:
            sw.option_gen_fair_cake()

        elif option == 4:
            sw.option_analyze_a_g6_file()

        elif option == 5:
            sw.option_analyze_multiple_g6()

        elif option == 6:
            sw.option_get_multiple_optimal()

        elif option == 7:
            sw.option_get_reliability_polynomial_optimal_graphs()

        elif option == 8:
            sw.option_graphs_comparison()

        elif option == 9:
            sw.option_compare_best_reliabilities()

        elif option == 10:
            sw.option_range_compare_best_reliabilities()

        elif option == 11:
            sw.option_compare_coefficients()

        elif option == 12:
            sw.option_gen_all_3ch_hamiltonian_opt()

        elif option == 13:
            sw.option_g6_files_data_analysis()

        elif option == 0:
            break

        else:
            print("Not a valid number")
