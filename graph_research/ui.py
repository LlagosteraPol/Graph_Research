from graph_research.core.ui_tools import *


def run():
    sw = Switcher()
    create_data_dirs()  # Create data folders if not exist

    while True:
        option = input_number \
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

             "13) Testing: Analyze the graphs inside a .g6 file and write results to database.\n"
             "14) Testing: Analyze the graphs inside a bunch of .g6 files and write results to database.\n"
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
            sw.option_g6_file_data_analysis2db()

        elif option == 14:
            sw.option_g6_files_data_analysis2db()

        elif option == 0:
            break

        else:
            print("Not a valid number")
