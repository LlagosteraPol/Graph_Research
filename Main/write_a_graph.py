from Main.graphtbox import *

def opt_ham_n7(chords):
    """
    This function retrieves the hamiltonian graph of 7 nodes and ch chords with optimal Reliability Polynomial
    :param chords: Number of chords of the hamiltonian graph
    :return: Networkx Hamiltonian graph of 7 nodes and ch chords
    """
    opt_c7 = nx.MultiGraph()
    opt_c7.add_edge(0, 1)
    opt_c7.add_edge(1, 2)
    opt_c7.add_edge(2, 3)
    opt_c7.add_edge(3, 4)
    opt_c7.add_edge(4, 5)
    opt_c7.add_edge(5, 6)
    opt_c7.add_edge(6, 0)
    if chords == 0: return opt_c7

    opt_c7.add_edge(0, 3)
    if chords == 1: return opt_c7
    opt_c7.add_edge(2, 5)
    if chords == 2: return opt_c7
    opt_c7.add_edge(1, 4)
    if chords == 3: return opt_c7
    opt_c7.add_edge(2, 6)
    if chords == 4: return opt_c7
    opt_c7.add_edge(0, 5)
    if chords == 5: return opt_c7
    opt_c7.add_edge(3, 6)
    if chords == 6: return opt_c7
    opt_c7.add_edge(1, 5)
    if chords == 7: return opt_c7

def opt_ham_n8(chords):
    """
    This function retrieves the hamiltonian graph of 8 nodes and ch chords with optimal Reliability Polynomial
    :param chords: Number of chords of the hamiltonian graph
    :return: Networkx Hamiltonian graph of 8 nodes and ch chords
    """
    opt_c8 = nx.MultiGraph()
    opt_c8.add_edge(0, 1)
    opt_c8.add_edge(1, 2)
    opt_c8.add_edge(2, 3)
    opt_c8.add_edge(3, 4)
    opt_c8.add_edge(4, 5)
    opt_c8.add_edge(5, 6)
    opt_c8.add_edge(6, 7)
    opt_c8.add_edge(7, 0)
    if chords == 0: return opt_c8

    opt_c8.add_edge(0, 4)
    if chords == 1: return opt_c8
    opt_c8.add_edge(2, 6)
    if chords == 2: return opt_c8
    opt_c8.add_edge(1, 5)
    if chords == 3: return opt_c8
    opt_c8.add_edge(3, 7)
    if chords == 4: return opt_c8

    opt_c8.add_edge(0, 3)
    if chords == 5: return opt_c8
    opt_c8.add_edge(4, 7)
    if chords == 6: return opt_c8
    opt_c8.add_edge(1, 6)
    if chords == 7: return opt_c8
    opt_c8.add_edge(2, 5)
    if chords == 8: return opt_c8

    opt_c8.add_edge(0, 5)
    if chords == 9: return opt_c8
    opt_c8.add_edge(1, 4)
    if chords == 10: return opt_c8
    opt_c8.add_edge(2, 7)
    if chords == 11: return opt_c8
    opt_c8.add_edge(3, 6)
    if chords == 12: return opt_c8

    opt_c8.add_edge(0, 2)
    if chords == 13: return opt_c8
    opt_c8.add_edge(4, 6)
    if chords == 14: return opt_c8
    opt_c8.add_edge(1, 3)
    if chords == 15: return opt_c8
    opt_c8.add_edge(5, 7)
    if chords == 16: return opt_c8

    opt_c8.add_edge(0, 6)
    if chords == 17: return opt_c8
    opt_c8.add_edge(2, 4)
    if chords == 18: return opt_c8
    opt_c8.add_edge(1, 7)
    if chords == 19: return opt_c8
    opt_c8.add_edge(3, 5)
    if chords == 20: return opt_c8

    else: return None

"""
g_list = list()
for i in range(0, 21):
    g_list.append(opt_ham_n8(i))

GraphTools.analyze_graphs(g_list, os.getcwd() + "/Data", "Optimal_Hamiltonian")
"""

p = sympy.symbols("p")
"""
n_7_1 = opt_ham_n7(5)
n_7_2 = copy.deepcopy(n_7_1)

n_7_1.add_edge(3, 6)
poly1 = GraphRel.relpoly_binary_basic(n_7_1)
print(list(n_7_1.edges))
print(poly1, " = ", poly1.subs({p: 0.6}))

n_7_2.add_edge(1, 3)
poly2 = GraphRel.relpoly_binary_basic(n_7_2)
print(list(n_7_2.edges))
print(poly2, " = ", poly2.subs({p: 0.6}))
"""

"""
for i in range(0, 8):
    print("------------N7 CH", i, "------------")
    n_7 = opt_ham_n7(i)
    n_7.add_edge(3, 6)
    poly1 = GraphRel.relpoly_binary_basic(n_7)
    print(list(n_7.edges))
    print(poly1, " = ", poly1.subs({p: 0.6}))
"""
"""
opt_ham_n8_e13 = nx.MultiGraph()
opt_ham_n8_e13.add_edge(0, 4)
opt_ham_n8_e13.add_edge(0, 6)
opt_ham_n8_e13.add_edge(0, 7)
opt_ham_n8_e13.add_edge(1, 4)
opt_ham_n8_e13.add_edge(1, 6)
opt_ham_n8_e13.add_edge(1, 7)
opt_ham_n8_e13.add_edge(2, 5)
opt_ham_n8_e13.add_edge(2, 6)
opt_ham_n8_e13.add_edge(2, 7)
opt_ham_n8_e13.add_edge(3, 5)
opt_ham_n8_e13.add_edge(3, 6)
opt_ham_n8_e13.add_edge(3, 7)
opt_ham_n8_e13.add_edge(4, 5)

test_ham_n8_e14_1 = copy.deepcopy(opt_ham_n8_e13)
test_ham_n8_e14_1.add_edge(3, 7)
poly1 = GraphRel.relpoly_binary_basic(test_ham_n8_e14_1)
print(list(test_ham_n8_e14_1.edges))
print(poly1, " = ", poly1.subs({p: 0.6}))

test_ham_n8_e14_2 = opt_ham_n8(6)
poly2 = GraphRel.relpoly_binary_basic(test_ham_n8_e14_2)
print(list(test_ham_n8_e14_2.edges))
print(poly2, " = ", poly2.subs({p: 0.6}))
"""
"""
opt_ham_n8_e14 = nx.MultiGraph()
opt_ham_n8_e14.add_edge(0, 4)
opt_ham_n8_e14.add_edge(0, 5)
opt_ham_n8_e14.add_edge(0, 6)
opt_ham_n8_e14.add_edge(0, 7)
opt_ham_n8_e14.add_edge(1, 4)
opt_ham_n8_e14.add_edge(1, 5)
opt_ham_n8_e14.add_edge(1, 6)
opt_ham_n8_e14.add_edge(1, 7)
opt_ham_n8_e14.add_edge(2, 4)
opt_ham_n8_e14.add_edge(2, 6)
opt_ham_n8_e14.add_edge(2, 7)
opt_ham_n8_e14.add_edge(3, 5)
opt_ham_n8_e14.add_edge(3, 6)
opt_ham_n8_e14.add_edge(3, 7)

ham_path_n8_e14 = GraphTools.hamilton_path(opt_ham_n8_e14)

print(ham_path_n8_e14)
"""
"""
opt = GraphRel.fair_cake_algorithm(7, 8)
poly = GraphRel.relpoly_binary_basic(opt)
print(list(opt.edges))
print(poly)
"""


"""
fc_6n = nx.Graph()
fc_6n.add_edges_from([(0, 1), (0, 3), (0, 5), (1, 2), (1, 4), (2, 3), (2, 5), (3, 4), (4, 5)])

fc_7n = GraphRel.fair_cake_algorithm(7, 3)

fc_8n = GraphRel.fair_cake_algorithm(8, 3)

fc_9n = GraphRel.fair_cake_algorithm(9, 3)

fc_10n = GraphRel.fair_cake_algorithm(10, 3)

fc_11n = GraphRel.fair_cake_algorithm(11, 3)

fc_10n_m = nx.Graph()
fc_10n_m.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
(0, 6), (4, 9), (7,2)])

fc_12n = nx.Graph()
fc_12n.add_edges_from([(0, 1), (0, 11), (1, 2), (1, 7), (2, 3), (3, 4), (3, 9), (4, 5), (5, 6), (5, 11), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
)

fc_13n = GraphRel.fair_cake_algorithm(13, 3)

fc_14n = GraphRel.fair_cake_algorithm(14, 3)

fc_16n = GraphRel.fair_cake_algorithm(16, 3)

fc_17n = GraphRel.fair_cake_algorithm(17, 3)

fc_18n = GraphRel.fair_cake_algorithm(18, 3)

fc_20n = GraphRel.fair_cake_algorithm(20, 3)

fc_22n = GraphRel.fair_cake_algorithm(22, 3)

fc_24n = nx.Graph()
fc_24n.add_edges_from([(0, 1), (0, 23), (1, 2), (2, 3), (3, 4), (3, 15), (4, 5), (5, 6), (6, 7), (7, 8), (7, 19), (8, 9), (9, 10), (10, 11), (11, 12), (11, 23), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23)]
)

fc_30n = GraphRel.fair_cake_algorithm(30, 3)
"""

#GraphRel.fair_cake_defiance(fc_17n, False)

#print("\n Sp trees method: ", GraphTools.spanning_trees_fair_cake(fc_30n))

#pol1 = GraphRel.relpoly_binary_improved(fc_10n)
#pol2 = GraphRel.relpoly_binary_improved(fc_10n_m)

#print("\nPoly 1: \n", pol1, " -> (p = ",0.6,") = ", pol1.subs({p: 0.6}))
#print("\nPoly 2: \n", pol2, " -> (p = ",0.6,") = ", pol2.subs({p: 0.6}))

#for i in range(1, 6):
#    GraphRel.fair_cake_defiance(GraphRel.fair_cake_algorithm(10, i), False)
"""
res = 0

for subset in itt.combinations([3,1,3,1], 3):
    aux = 1
    for i in subset:
        aux *= i
    res += aux

for subset in itt.combinations([2,2,2,2], 3):
    aux = 1
    for i in subset:
        aux *= i
    res += aux

for subset in itt.combinations([3,1,3,1], 3):
    aux = 1
    for i in subset:
        aux *= i
    res += aux


for subset in itt.combinations([1,1,2,1,1,2], 4):
    print(subset)
    aux = 1
    for i in subset:
        aux *= i
    res += aux

print("Total", res)

k = sympy.symbols("k")

sp_poly_6k4 = GraphTools.spanning_trees_polynomial((k + 1), (k + 1), k, (k + 1), (k + 1), k )
sp_poly_m_6k4 = GraphTools.spanning_trees_polynomial( (k + 1), k, (k + 1), (k + 1), (k + 1), k )

print(sympy.simplify(sp_poly_6k4))
print("Spanning trees 6k+4:", sp_poly_6k4.subs({k: 2}), "\n")
print(sympy.simplify(sp_poly_m_6k4))
print("Spanning trees 6k+4:", sp_poly_m_6k4.subs({k: 2}), "\n")


sp_poly_6k2 = GraphTools.spanning_trees_polynomial(k , (k + 1), k, k, (k + 1), k )
sp_poly_m_6k2 = GraphTools.spanning_trees_polynomial((k + 1), k ,  k, k, (k + 1), k )

print(sympy.simplify(sp_poly_6k2))
print("Spanning trees 6k+2:", sp_poly_6k2.subs({k: 1}), "\n")
print(sympy.simplify(sp_poly_m_6k2))
print("Spanning trees 6k+2:", sp_poly_m_6k2.subs({k: 1}), "\n")
"""

"""
path = os.getcwd() + "/Data/Graph6/"

# 2 chords

opt_n6_e8 = nx.Graph([(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5)])

opt_n7_e9 = nx.Graph([(0, 3), (0, 4), (0, 6), (1, 4), (1, 5), (1, 6), (2, 5), (2, 6), (3, 5)])

opt_n8_e10 = nx.Graph([(0, 4), (0, 5), (1, 4), (1, 7), (2, 5), (2, 6), (3, 6), (3, 7), (4, 6), (5, 7)])

opt_n9_e11 = nx.Graph([(0, 4), (0, 7), (1, 5), (1, 6), (2, 5), (2, 8), (3, 6), (3, 7), (4, 8), (5, 7), (6, 8)])

opt_n10_e12 = nx.Graph([(0, 4), (0, 6), (0, 8), (1, 5), (1, 8), (2, 6), (2, 7), (3, 7), (3, 8), (4, 9), (5, 9), (7, 9)])

opt_n11_e13 = nx.Graph([(0, 5), (0, 9), (1, 6), (1, 7), (1, 10), (2, 6), (2, 8), (2, 9), (3, 7), (3, 9), (4, 8), (4, 10), (5, 10)])


# 3 Chords

opt_n6_e9 = nx.Graph([(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)])

opt_n7_e10 = nx.Graph([(0, 3), (0, 4), (0, 6), (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5)])

opt_n8_e11 = nx.Graph([(0, 3), (0, 5), (1, 4), (1, 5), (1, 7), (2, 5), (2, 6), (2, 7), (3, 6), (3, 7), (4, 6)])

opt_n9_e12 = nx.Graph([(0, 3), (0, 6), (1, 4), (1, 6), (1, 8), (2, 5), (2, 6), (2, 7), (3, 7), (3, 8), (4, 7), (5, 8)])

opt_n10_e13 = nx.Graph([(0, 4), (0, 7), (0, 9), (1, 5), (1, 7), (1, 8), (2, 6), (2, 7), (3, 6), (3, 8), (4, 8), (5, 9), (6, 9)])

opt_n11_e14 = nx.Graph([(0, 5), (0, 8), (1, 6), (1, 9), (2, 6), (2, 10), (3, 7), (3, 8), (4, 7), (4, 9), (5, 9), (5, 10), (6, 8), (7, 10)])

#opt_n12_e15 = nx.Graph()


nx.write_graph6(opt_n6_e8, path + "Optimal_n6_e8.g6")
nx.write_graph6(opt_n7_e9, path + "Optimal_n7_e9.g6")
nx.write_graph6(opt_n8_e10, path + "Optimal_n8_e10.g6")
nx.write_graph6(opt_n9_e11, path + "Optimal_n9_e11.g6")
nx.write_graph6(opt_n10_e12, path + "Optimal_n10_e12.g6")
nx.write_graph6(opt_n11_e13, path + "Optimal_n11_e13.g6")
"""
"""
prob = 0.6

fc_t = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
                 (0, 5), (2, 7), (4, 9)])

mod_t = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
                  (0, 5), (2, 7), (3, 9)])

opt_t = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
                  (0, 4), (1, 6), (3, 8)])

fct_rel = GraphRel.relpoly_binary_improved(fc_t)
mod_rel = GraphRel.relpoly_binary_improved(mod_t)
opt_rel = GraphRel.relpoly_binary_improved(opt_t)

print("\nOpt Rel\n", opt_rel, "\n= ", opt_rel.subs({p: prob}))
print("\nFC Rel:     \n", fct_rel, "\n= ", fct_rel.subs({p: prob}))
print("\nMod Rel:     \n", mod_rel, "\n= ", mod_rel.subs({p: prob}))
"""
"""
for i in range(8, 35, 6):
    print("Constructing file: Opt_Hamilton_n" + str(i)+"_ch" + str(3))
    g_list = GraphTools.gen_all_3ch_hamiltonian_opt(i)
    GraphTools.gen_g6_file(g_list, "Opt_Hamilton_n" + str(i)+"_ch" + str(3))
"""
#g_list = GraphTools.gen_all_3ch_hamiltonian_opt(10)
#GraphTools.gen_g6_file(g_list, "Opt_Hamilton_n" + str(10)+"_ch" + str(3))
#g_list = GraphTools.gen_all_hamiltonian(10, 3)

# C
#g = nx.MultiGraph([(0,1), (0,2), (1,2),(1,3), (2,3)])
#GraphTools.plot(g)

# C
#g2 = nx.MultiGraph([(0,1), (0,2), (1,2),(2,3), (2,4), (3,4)])
#GraphTools.plot(g2)

# C
#g3 = nx.MultiGraph([(0,1), (0,3), (1,4), (1,2), (2,3), (3,5), (4,5), (4,6), (5,6)])
#GraphTools.plot(g3)

# c
#g4 = nx.MultiGraph([(0,1), (0,3), (1,4), (1,2), (2,3), (3,5), (4,5), (4,6), (5,6)])
#GraphTools.plot(g4)

#Benchmarks.relpoly_ordered_cycles_console_benchmark([g3])

#hams = GraphTools.gen_all_hamiltonian(10, 20)

#Benchmarks.relpoly_binary_improved_console_benchmark(hams, 100)
"""
n7e12 = nx.Graph([(0, 3), (0, 4), (0, 5), (0, 6), (1, 3), (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 6)]
)
n8e12 = nx.Graph([(0, 3), (0, 4), (0, 5), (1, 4), (1, 5), (1, 6), (2, 5), (2, 6), (2, 7), (3, 6), (3, 7), (4, 7)]
)
n10e15 = nx.Graph([(0, 3), (0, 6), (0, 9), (1, 4), (1, 6), (1, 8), (2, 5), (2, 6), (2, 7), (3, 7), (3, 8), (4, 7), (4, 9), (5, 8), (5, 9)]
)
n10e17 = nx.Graph([(0, 8), (0, 9), (1, 8), (1, 9), (2, 8), (2, 9), (3, 8), (3, 9), (4, 8), (4, 9), (5, 8), (5, 9), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
)
n21e24 = nx.Graph([(0, 1), (0, 10), (0, 20), (1, 2), (2, 3), (3, 4), (3, 14), (4, 5), (5, 6), (6, 7), (7, 8), (7, 17), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20)]
)

time_start = time.process_time()
pol = Utilities.polynomial2binomial(GraphRel.relpoly_binary_improved(n7e12))
time_elapsed = (time.process_time() - time_start)
print("Reliability improved: ", time_elapsed)


time_start = time.process_time()
connections = GraphTools.get_all_connected_graphs(n7e12)
n_g = list()
for element in connections.values():
    n_g.append(len(element))
time_elapsed = (time.process_time() - time_start)
print("Connected components: ", time_elapsed)
"""
"""
check1 = nx.Graph([(0, 5), (0, 9), (1, 6), (1, 7), (1, 10), (2, 6), (2, 8), (2, 9), (3, 7), (3, 9), (4, 8), (4, 10), (5, 10)])
check2 = nx.Graph([(0, 5), (0, 6), (0, 10), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (3, 9), (4, 9), (4, 10), (5, 10), (6, 10)])

check3 = nx.Graph([(0, 6), (0, 9), (1, 7), (1, 8), (2, 7), (2, 9), (3, 7), (3, 10), (4, 8), (4, 9), (5, 8), (5, 10), (6, 10)])
check4 = nx.Graph([(0, 8), (0, 9), (0, 10), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 9), (8, 10), (9, 10)])

rel1 = GraphRel.relpoly_binary_improved(check1)
print("H_B: ", Utilities.polynomial2binomial(rel1))

rel2 = GraphRel.relpoly_binary_improved(check2)
print("H_W: ", Utilities.polynomial2binomial(rel2))

rel3 = GraphRel.relpoly_binary_improved(check3)
print("G_B: ", Utilities.polynomial2binomial(rel3))

rel4 = GraphRel.relpoly_binary_improved(check4)
print("G_W: ", Utilities.polynomial2binomial(rel4))
"""
"""
out_path = os.getcwd() + "/Data/Graph6/"
for n in range(11, 12):
    GraphTools.gen_all_3ch_hamiltonian_opt(n, out_path)
"""



"""
pd1 = parse_expr("Poly(2668.0*p**28 - 11836.0*p**27 + 19732.0*p**26 - 14652.0*p**25 + 4089.0*p**24 + 2.8421709430404e-14*p**7 - "
                 "1.4210854715202e-13*p**6 + 2.8421709430404e-13*p**5 - 2.8421709430404e-13*p**4 + 1.4210854715202e-13*p**3 - 2.8421709430404e-14*p**2, p, domain='RR')")

pd2 = sympy.Poly(2668.0*p**28 - 11836.0*p**27 + 19732.0*p**26 - 14652.0*p**25 + 4089.0*p**24 + 2.8421709430404e-14*p**7 - 1.4210854715202e-13*p**6 + 2.8421709430404e-13*p**5 - 2.8421709430404e-13*p**4 + 1.4210854715202e-13*p**3 - 2.8421709430404e-14*p**2, p, domain='RR')


test = sympy.Poly(2668.0*p**28 - 11836.0*p**27 + 19732.0*p**26 - 14652.0*p**25 + 4089.0*p**24, p, domain='RR')
t, tt = Utilities.polynomial2binomial(test)
print(t)
pf1, nope = Utilities.polynomial2binomial(pd1)
print(pf1)

pd3 = sympy.poly(pf1)

if pd3 == test:
    print("True")

else:
    print("False")
"""

"""
g6 = b'XPS?GH?_???O?C?A?G?@??A????????@C??I??DG?E????g?@?B'
g = nx.from_graph6_bytes(g6)

polynomial = GraphRel.relpoly_binary_improved(g, 0)

print(polynomial)
"""

engine = db.create_engine('sqlite:///' + os.getcwd() + "/Data/DDBB/Graphs_DB.db", echo=False)
Session = db.orm.session.sessionmaker(bind=engine)
session = Session()
session._model_changes = {}
metadata = db.MetaData()
graphs = db.Table('Graphs', metadata, autoload=True, autoload_with=engine)

#df = DButilities.read_table(session, Table_Graph, conditions="nodes <= 8")

"""
# Migrate table
query = db.select([graphs])
df = pd.read_sql_query(query, engine)
df.rename(columns={'g6_id': 'g6'}, inplace=True)

df = df.astype({Table_Graph.g6_hash.name: Table_Graph.g6_hash.type.python_type,
                Table_Graph.g6.name: Table_Graph.g6.type.python_type,
                Table_Graph.nodes.name: Table_Graph.nodes.type.python_type,
                Table_Graph.edges.name: Table_Graph.edges.type.python_type,
                Table_Graph.hamiltonian.name: Table_Graph.hamiltonian.type.python_type,
                Table_Graph.hamiltonian_cycle.name: Table_Graph.hamiltonian_cycle.type.python_type,
                Table_Graph.graph_edges.name: Table_Graph.graph_edges.type.python_type,
                Table_Graph.avg_polynomial.name: Table_Graph.avg_polynomial.type.python_type,
                Table_Graph.polynomial.name: Table_Graph.polynomial.type.python_type,
                Table_Graph.spanning_trees.name: Table_Graph.spanning_trees.type.python_type,
                Table_Graph.edge_connectivity.name: Table_Graph.edge_connectivity.type.python_type,
                Table_Graph.min_k2_edge_cuts.name: Table_Graph.min_k2_edge_cuts.type.python_type,
                Table_Graph.automorphisms.name: Table_Graph.automorphisms.type.python_type,
                Table_Graph.diameter.name: Table_Graph.diameter.type.python_type,
                Table_Graph.probability_01.name: Table_Graph.probability_01.type.python_type,
                Table_Graph.probability_02.name: Table_Graph.probability_02.type.python_type,
                Table_Graph.probability_03.name: Table_Graph.probability_03.type.python_type,
                Table_Graph.probability_04.name: Table_Graph.probability_04.type.python_type,
                Table_Graph.probability_05.name: Table_Graph.probability_05.type.python_type,
                Table_Graph.probability_06.name: Table_Graph.probability_06.type.python_type,
                Table_Graph.probability_07.name: Table_Graph.probability_07.type.python_type,
                Table_Graph.probability_08.name: Table_Graph.probability_08.type.python_type,
                Table_Graph.probability_09.name: Table_Graph.probability_09.type.python_type})
df.set_index('g6_hash', inplace=True)


GraphTools.data_print(df, FormatType.SQL, os.getcwd() + "/Data/DDBB/" + "Graphs_DB_2")
"""
"""
# Add hash
for i in range(6, 12):
    print("Updating graph with ", i, " nodes")
    query = db.select([graphs]).where(graphs.columns.nodes == i)
    df = pd.read_sql_query(query, engine)
    for key, values in df.iterrows():
        g6 = values['g6_id'].rstrip().encode()
        hashed = hashlib.md5(g6)
        df.set_value(key, 'g6_hash', hashed.hexdigest())
    df.set_index('g6_id', inplace=True)
    DButilities.add_or_update(session, df, Table_Graph)
"""

#DButilities.add_column(session, 'Graphs', 'g6_hash' ,'VARCHAR')

"""
# Update polynomials to its binomial form
for i in range(25, 31):
    print("Updating graph with ", i, " nodes")
    query = db.select([graphs]).where(graphs.columns.nodes == i)
    df = pd.read_sql_query(query, engine)
    for key, values in df.iterrows():
        try:
            polynomial = Utilities.polynomial2binomial(sympy.Poly(values['polynomial']))[0]
        except:
            polynomial = 0
            print(values['g6'])
        df.set_value(key, 'polynomial', str(polynomial))
    df.set_index('g6_hash', inplace=True)
    DButilities.add_or_update(session, df, Table_Graph)
"""
"""
# Recalculate polynomials
for i in range(11, 25):
    print("Updating graph with ", i, " nodes")
    query = db.select([graphs]).where(graphs.columns.nodes == i)
    df = pd.read_sql_query(query, engine)
    for key, values in df.iterrows():
        g = nx.from_graph6_bytes(values['g6'].rstrip().encode())
        try:
            polynomial = Utilities.polynomial2binomial(GraphRel.relpoly_binary_improved(g))[0]
        except:
            polynomial = 0
            print(values['g6'])

        df.set_value(key, 'polynomial', str(polynomial))
    df.set_index('g6_hash', inplace=True)
    DButilities.add_or_update(session, df, Table_Graph)

session.close()
"""
"""
query = db.select([graphs]).where(graphs.columns.g6_hash == 'cd78506ebf81b3af64771c04e9013a70')
df = pd.read_sql_query(query, engine)

for key, values in df.iterrows():
    g = nx.from_graph6_bytes(values['g6'].rstrip().encode())
    try:
        polynomial = Utilities.polynomial2binomial(GraphRel.relpoly_binary_improved(g))[0]
    except:
        polynomial = 0
        print(values['g6'])

    df.set_value(key, 'polynomial', str(polynomial))

df.set_index('g6_hash', inplace=True)
DButilities.add_or_update(session, df, Table_Graph)
session.close()
"""
"""
test = pd.DataFrame({'g6_id': ['test'], 'nodes': [np.nan], 'edges': [np.nan], 'hamiltonian': [np.nan],
                     'hamiltonian_cycle': [np.nan], 'graph_edges': [np.nan], 'avg_polynomial': [np.nan],
                     'polynomial': [np.nan], 'spanning_trees': [np.nan], 'edge_connectivity': [np.nan],
                     'min_k2_edge_cuts': [np.nan], 'automorphisms': [np.nan], 'diameter': [np.nan],
                     'probability_01': [np.nan], 'probability_02': [np.nan], 'probability_02': [np.nan],
                     'probability_03': [np.nan], 'probability_04': [np.nan], 'probability_05': [np.nan],
                     'probability_06': [np.nan], 'probability_07': [np.nan], 'probability_08': [np.nan],
                     'probability_09': [np.nan]})
#test.set_index('g6_id')

test2 = pd.DataFrame({'g6_id': ['test2'], 'nodes': [np.nan], 'edges': [np.nan], 'hamiltonian': [np.nan],
                     'hamiltonian_cycle': [np.nan], 'graph_edges': [np.nan], 'avg_polynomial': [np.nan],
                     'polynomial': [np.nan], 'spanning_trees': [np.nan], 'edge_connectivity': [np.nan],
                     'min_k2_edge_cuts': [np.nan], 'automorphisms': [np.nan], 'diameter': [np.nan],
                     'probability_01': [np.nan], 'probability_02': [np.nan], 'probability_02': [np.nan],
                     'probability_03': [np.nan], 'probability_04': [np.nan], 'probability_05': [np.nan],
                     'probability_06': [np.nan], 'probability_07': [np.nan], 'probability_08': [np.nan],
                     'probability_09': [np.nan]})
#test2.set_index('g6_id')

tests = test.append(test2)
tests.set_index('g6_id', inplace=True)

test_dict = test.to_dict(orient="records")
test_dict2 = test.to_dict(orient="index")

tests_dict = tests.to_dict(orient="records")
tests_dict2 = tests.to_dict(orient="index")
DButilities.add_or_update(session, tests, Table_Graph)
"""

print(Utilities.polynomial2binomial(sympy.Poly(72.0*p**13 - 702.0*p**12 + 3024.0*p**11 - 7532.0*p**10 + 11893.0*p**9 - 12228.0*p**8 + 8027.0*p**7 - 3093.0*p**6 + 540.0*p**5, p, domain='RR')))

print(Utilities.polynomial2binomial(sympy.Poly(4.0*p**7 - 11.0*p**6 + 8.0*p**5, p, domain='RR')))















