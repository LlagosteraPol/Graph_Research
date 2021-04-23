import networkx as nx


class GraphsStorage(object):
    """
    Bunch of methods to generate different types of networkx graphs
    """

    @staticmethod
    def print_options():
        print("\nChoose a graph to calculate its reliability: ")
        print("\t0- Default")
        print("\t1- Hypercube n3")
        print("\t2- Tree n4")
        print("\t3- Loop")
        print("\t4- Cycle n3")
        print("\t5- Square")
        print("\t6- Graph n=3, e=3; 0 cycle, 1 bridges, 0 self loop")
        print("\t7- Graph n=2, e=3; 1 cycle, 0 bridges, 1 self loop")
        print("\t8- Graph n=4, e=4; 1 cycle, 1 bridge, 0 self loops")
        print("\t9- Graph n=3, e=4; 1 cycle, 2 bridges, 0 self loops")
        print("\t10- Graph n=2, e=2; 1 bridge, 1 self loop")
        print("\t11- Graph n=1, e=2; 2 self loop")

    def __represents_int(self, s):
        try:
            int(s)
            return True

        except ValueError:
            return False

    def switch_graph(self, graph):

        if self.__represents_int(graph):

            if graph is 0:
                option = getattr(self, "default", lambda: "Graph not existent")
            elif graph is 1:
                option = getattr(self, "hypercube_n3", lambda: "Graph not existent")
            elif graph is 2:
                option = getattr(self, "tree_n4", lambda: "Graph not existent")
            elif graph is 3:
                option = getattr(self, "loop", lambda: "Graph not existent")
            elif graph is 4:
                option = getattr(self, "cycle_n3", lambda: "Graph not existent")
            elif graph is 5:
                option = getattr(self, "square", lambda: "Graph not existent")
            elif graph is 6:
                option = getattr(self, "graph1", lambda: "Graph not existent")
            elif graph is 7:
                option = getattr(self, "graph2", lambda: "Graph not existent")
            elif graph is 8:
                option = getattr(self, "graph3", lambda: "Graph not existent")
            elif graph is 9:
                option = getattr(self, "graph4", lambda: "Graph not existent")
            elif graph is 10:
                option = getattr(self, "graph5", lambda: "Graph not existent")
            elif graph is 11:
                option = getattr(self, "graph6", lambda: "Graph not existent")

        else:
            option = getattr(self, graph, lambda: "Graph not existent")

        return option()

    @staticmethod
    def default():
        g = nx.MultiGraph()

        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        return g

    @staticmethod
    def hypercube_n3():
        q3 = nx.MultiGraph()

        q3.add_edge(0, 1)
        q3.add_edge(0, 3)
        q3.add_edge(0, 4)
        q3.add_edge(1, 2)
        q3.add_edge(1, 5)
        q3.add_edge(2, 3)
        q3.add_edge(2, 6)
        q3.add_edge(3, 7)
        q3.add_edge(4, 5)
        q3.add_edge(4, 7)
        q3.add_edge(5, 6)
        q3.add_edge(6, 7)

        return q3

    @staticmethod
    def tree_n4():
        t = nx.MultiGraph()

        t.add_edge(0, 1, weight=0.1)
        t.add_edge(1, 2, weight=0.2)
        t.add_edge(2, 3, weight=0.3)
        t.add_edge(3, 4, weight=0.4)

        return t

    @staticmethod
    def loop():
        l = nx.MultiGraph()

        l.add_edge(0, 0, weight=0.1)
        l.add_edge(0, 0, weight=0.2)
        l.add_edge(0, 0, weight=0.3)
        l.add_edge(0, 0, weight=0.4)

        return l

    @staticmethod
    def cycle_n3():
        c3 = nx.MultiGraph()

        c3.add_edge(0, 1)
        c3.add_edge(1, 2)
        c3.add_edge(2, 0)

        return c3

    @staticmethod
    def square():
        sq = nx.MultiGraph()

        sq.add_edge(0, 1)
        sq.add_edge(1, 2)
        sq.add_edge(2, 3)
        sq.add_edge(3, 0)

        return sq

    @staticmethod
    def graph1():
        g1 = nx.MultiGraph()

        g1.add_edge(0, 3)
        g1.add_edge(0, 3)
        g1.add_edge(0, 1)

        return g1

    @staticmethod
    def graph2():
        g2 = nx.MultiGraph()

        g2.add_edge(0, 3)
        g2.add_edge(0, 3)
        g2.add_edge(0, 0)

        return g2

    @staticmethod
    def graph3():
        g3 = nx.MultiGraph()

        g3.add_edge(0, 3)
        g3.add_edge(0, 2)
        g3.add_edge(2, 1)
        g3.add_edge(3, 2)
        g3.add_edge(2, 1)

        return g3

    @staticmethod
    def graph4():
        g4 = nx.MultiGraph()

        g4.add_edge(0, 3)
        g4.add_edge(0, 2)
        g4.add_edge(0, 2)
        g4.add_edge(3, 2)

        return g4

    @staticmethod
    def graph5():
        g5 = nx.MultiGraph()

        g5.add_edge(0, 1)
        g5.add_edge(0, 0)

        return g5

    @staticmethod
    def graph6():
        g6 = nx.MultiGraph()

        g6.add_edge(0, 0)
        g6.add_edge(0, 0)

        return g6
