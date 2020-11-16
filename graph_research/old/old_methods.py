from graph_research.core.graphtbox import *

class OldMethods(object):

    @staticmethod
    def _recursive_improved_old(g, debug=False):
        """
        This is the improved contraction-deletion algorithm. In each recursion, if there exist some method
        that can retrieve the Reliability Polynomial directly or with less cost than another recursion,
        will retrieve it and stop the recursion in that generated sub-graph.
        :param g: Networkx graph
        :return: The reliability Polynomial of the given graph or another execution of the method.
        """
        p = sympy.symbols('p')
        # Get information about the graph
        bal_pedges, tree_edges, cycle_edges, total_edges = GraphTools.get_detailed_graph_edges(g)

        # If the graph is not connected, then it has a rel poly of 0
        if not nx.is_connected(g):
            return sympy.Poly(0, p)

        # if # edges > 0, then we perform the two subcases of the Factoring theorem.
        if len(g.edges()) > 0:
            # If the graph is a tree
            if tree_edges != {} and cycle_edges == []:
                return GraphRel.relpoly_multitree(bal_pedges, tree_edges)

            # If the graph only has one cycle
            elif tree_edges == {} and len(cycle_edges) == 1:
                return GraphRel.relpoly_multicycle(bal_pedges, cycle_edges[0])

            else:
                # Look for joined cycles
                # common_edge = GraphTools.get_a_common_edge(g)
                common_edge = choice(list(g.edges()))

                # If there aren't joined cycles try to apply direct formula
                if common_edge is None or len(common_edge) < 2:
                    if bal_pedges[1] == 1:  # If the graph has no multiedges
                        return GraphRel.relpoly_treecyc(g, cycle_edges)

                    # Else is a multiedge. If is a tree+cycles graph apply the properly formula
                    return GraphRel.relpoly_multi_treecyc(bal_pedges, tree_edges, cycle_edges, g)
                    # e = choice(list(g.edges()))  # Use in case the previous return is deleted

                else:
                    """
                    if bal_pedges[1] == 1 and debug:  # If the graph has no multiedges
                        # if len(cycle_edges) == 2: # If the graph has only two fused cycles
                        #    return GraphRel.relpoly_two_fused_cycles(g, cycle_edges)
                        if tree_edges == {}:
                            return GraphRel.relpoly_ordered_cycles(g, cycle_edges)
                    """
                    e = copy.deepcopy(common_edge)

                contracted = nx.contracted_edge(g, e, self_loops=False)  # TODO: Expected tuple
                g.remove_edge(*e)
                rec_deleted = GraphRel._recursive_improved_old(g)

                rec_contracted = GraphRel._recursive_improved_old(contracted)

                s = sympy.Poly(p) * rec_contracted + sympy.Poly(1 - p) * rec_deleted

                return s

        # Otherwise, we only have 0 edges and 1 vertex, which is connected, so we return 1.
        return sympy.Poly(1, p)