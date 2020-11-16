import os
import networkx as nx
#My classes
from .enums import *

class UITools(object):

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