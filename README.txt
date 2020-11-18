=================
GraphResearch Readme
=================
-------------------------
Realiability Focused Tools
-------------------------

Introduction
============

This project provides several algorithms and methods that create, process, and analyze graphs. Although the project is
focused on the Reliability Polynomial, it also gives useful tools to work with graphs.

Internally, the format of the graphs used inside the methods is Networkx graph objects. Externally, in order to write
and read them, the most used format is graph6 stored into '.g6' files. The resulting data of the analysis is stored
in .txt or into a SQLite database called 'Graphs_DB.db'. If the output format is desired to be another, in 'graphbox.py'
file there is the method 'data_print()' which stores the given data (dataframe) into CSV, JSON, Html or excel
(experimental), as well as SQLite.


Setup and running
-----------------

Some methods, in order to work, they need the following folder structure (inside graph_research folder) to read and
write data:
* data
    * databases
    * graph6
    * plain_results
    * test_results
    * tmp

This structure can be created in 3 ways; manually, running the method 'create_data_dirs()' inside ui_tools.py,
running '__main__.py' file.

In order to provide an easy insight of the functionality of the project. It has been implemented a very simple UI that
executes some of the methods provided in the project. To run the UI, simply run the file '__main__.py'.


Author
------
Pol Llagostera Blasco
<mailto:pol.llagostera@udl.cat>