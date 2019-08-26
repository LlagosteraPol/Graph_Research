from Main.graphtbox import *

"""
g = nx.MultiGraph()

g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 0)
g.add_edge(3, 4)
g.add_edge(3, 5)
g.add_edge(3, 6)
g.add_edge(3, 7)
g.add_edge(3, 8)
g.add_edge(3, 9)
g.add_edge(4, 5)
g.add_edge(6, 7)
g.add_edge(8, 11)
g.add_edge(9, 10)
g.add_edge(10, 11)



k = nx.MultiGraph()

k.add_edge(0, 1)
k.add_edge(1, 2)
k.add_edge(2, 3)
k.add_edge(3, 0)
k.add_edge(2, 4)
k.add_edge(2, 5)
k.add_edge(2, 6)
k.add_edge(2, 7)
k.add_edge(4, 5)
k.add_edge(6, 7)


f = nx.MultiGraph()

f.add_edge(0, 1)
f.add_edge(0, 2)
f.add_edge(2, 1)
f.add_edge(1, 3)
f.add_edge(3, 4)
f.add_edge(3, 5)
f.add_edge(5, 6)
f.add_edge(5, 8)
f.add_edge(6, 7)
f.add_edge(7, 8)

z = nx.MultiGraph()

z.add_edge(0, 1)
z.add_edge(0, 2)
z.add_edge(1, 3)
z.add_edge(1, 4)
z.add_edge(2, 3)
z.add_edge(3, 4)


treecyc = nx.MultiGraph()

treecyc.add_edge(0, 1)
treecyc.add_edge(0, 2)
treecyc.add_edge(1, 2)
treecyc.add_edge(1, 3)
treecyc.add_edge(3, 4)
treecyc.add_edge(3, 5)
treecyc.add_edge(5, 6)
treecyc.add_edge(5, 8)
treecyc.add_edge(6, 7)
treecyc.add_edge(7, 8)


n11e13 = nx.MultiGraph()

n11e13.add_edge(0, 1)
n11e13.add_edge(0, 2)
n11e13.add_edge(1, 2)
n11e13.add_edge(2, 3)
n11e13.add_edge(2, 5)
n11e13.add_edge(3, 4)
n11e13.add_edge(4, 5)
n11e13.add_edge(5, 6)
n11e13.add_edge(5, 10)
n11e13.add_edge(6, 7)
n11e13.add_edge(7, 8)
n11e13.add_edge(8, 9)
n11e13.add_edge(9, 10)

n4e8 = nx.MultiGraph()
n4e8.add_edge(0, 1)
n4e8.add_edge(0, 1)
n4e8.add_edge(1, 2)
n4e8.add_edge(1, 2)
n4e8.add_edge(1, 2)
n4e8.add_edge(2, 3)
n4e8.add_edge(2, 3)
n4e8.add_edge(0, 3)

n3e6 = nx.MultiGraph()
n3e6.add_edge(0, 1)
n3e6.add_edge(0, 1)
n3e6.add_edge(1, 2)
n3e6.add_edge(2, 0)
n3e6.add_edge(2, 0)
n3e6.add_edge(2, 0)

square_2pe = nx.MultiGraph()
square_2pe.add_edge(0, 1)
square_2pe.add_edge(0, 1)
square_2pe.add_edge(1, 2)
square_2pe.add_edge(1, 2)
square_2pe.add_edge(2, 3)
square_2pe.add_edge(2, 3)
square_2pe.add_edge(3, 0)
square_2pe.add_edge(3, 0)

#First graph
fgraph = nx.MultiGraph()
fgraph.add_edge(0, 1)
fgraph.add_edge(0, 1)
fgraph.add_edge(0, 2)
fgraph.add_edge(2, 3)
fgraph.add_edge(3, 0)
fgraph.add_edge(3, 0)
fgraph.add_edge(3, 0)
fgraph.add_edge(3, 0)
fgraph.add_edge(1, 3)
fgraph.add_edge(1, 3)
fgraph.add_edge(1, 3)
fgraph.add_edge(3, 4)
fgraph.add_edge(3, 4)
fgraph.add_edge(3, 4)
fgraph.add_edge(4, 5)
fgraph.add_edge(4, 5)
fgraph.add_edge(5, 6)

fgraph.add_edge(6, 7)
fgraph.add_edge(6, 7)
fgraph.add_edge(6, 7)
fgraph.add_edge(7, 8)
fgraph.add_edge(7, 8)
fgraph.add_edge(8, 6)

fgraph.add_edge(8, 9)
fgraph.add_edge(9, 10)
fgraph.add_edge(9, 10)

#First graph
s_tree = nx.MultiGraph()
s_tree.add_edge(0, 1)
s_tree.add_edge(0, 1)
s_tree.add_edge(0, 1)
s_tree.add_edge(1, 2)
s_tree.add_edge(1, 2)
s_tree.add_edge(2, 3)

# CodeBreaker 1 Graph
c_breaker = nx.MultiGraph()
c_breaker.add_edge(0, 1)
c_breaker.add_edge(1, 3)

c_breaker.add_edge(1, 6)

c_breaker.add_edge(3, 4)
c_breaker.add_edge(4, 5)

c_breaker.add_edge(6, 7)
c_breaker.add_edge(7, 8)
c_breaker.add_edge(8, 9)
c_breaker.add_edge(9, 10)
c_breaker.add_edge(10, 5)

c_breaker.add_edge(5, 11)
c_breaker.add_edge(5, 13)
c_breaker.add_edge(5, 14)

c_breaker.add_edge(11, 12)
c_breaker.add_edge(12, 13)
c_breaker.add_edge(13, 14)


# CodeBreaker 2 Graph
c_breaker2 = nx.MultiGraph()

c_breaker2.add_edge(0, 1)
c_breaker2.add_edge(1, 3)

c_breaker2.add_edge(1, 2)
c_breaker2.add_edge(1, 4)
c_breaker2.add_edge(1, 4)
c_breaker2.add_edge(1, 4)

c_breaker2.add_edge(2, 5)

c_breaker2.add_edge(3, 4)

c_breaker2.add_edge(4, 5)

# CodeBreaker 2 Graph
c_breaker3 = nx.MultiGraph()
c_breaker3.add_edge(2, 10)
c_breaker3.add_edge(2, 8)
c_breaker3.add_edge(4, 5)
c_breaker3.add_edge(4, 6)
c_breaker3.add_edge(8, 10)
c_breaker3.add_edge(8, 12)
c_breaker3.add_edge(8, 3)
c_breaker3.add_edge(8, 5)
c_breaker3.add_edge(3, 11)
c_breaker3.add_edge(6, 7)
c_breaker3.add_edge(6, 14)
c_breaker3.add_edge(10, 11)
c_breaker3.add_edge(12, 13)
c_breaker3.add_edge(13, 15)
c_breaker3.add_edge(14, 15)

#Core Graph
multi_core = nx.MultiGraph()
multi_core.add_edge(0,1)
multi_core.add_edge(0,1)
multi_core.add_edge(0,2)
multi_core.add_edge(0,2)
multi_core.add_edge(0,2)
multi_core.add_edge(0,3)
multi_core.add_edge(0,4)
multi_core.add_edge(1,3)
multi_core.add_edge(1,2)
multi_core.add_edge(2,3)
multi_core.add_edge(2,4)


#Core Graph
core = nx.MultiGraph()
core.add_edge(0,1)
core.add_edge(0,2)
core.add_edge(0,3)
core.add_edge(0,5)
core.add_edge(0,6)
core.add_edge(1,2)
core.add_edge(1,3)
core.add_edge(1,4)
core.add_edge(1,7)
core.add_edge(2,4)
core.add_edge(2,5)
core.add_edge(2,8)
core.add_edge(3,7)
core.add_edge(3,6)
core.add_edge(4,7)
core.add_edge(4,8)
core.add_edge(5,6)
core.add_edge(5,8)


# Hypercube
hypercube = nx.MultiGraph()
hypercube.add_edge(0, 1)
hypercube.add_edge(0, 2)
hypercube.add_edge(0, 4)

hypercube.add_edge(1, 5)

hypercube.add_edge(2, 3)
hypercube.add_edge(2, 6)

hypercube.add_edge(3, 1)
hypercube.add_edge(3, 7)

hypercube.add_edge(4, 6)
hypercube.add_edge(4, 5)

hypercube.add_edge(5, 7)

hypercube.add_edge(6, 7)

# Hypercube_3

hypercube3 = nx.MultiGraph()
hypercube3.add_edge(0,1)
hypercube3.add_edge(0,2)
hypercube3.add_edge(0,4)
hypercube3.add_edge(0,8)

hypercube3.add_edge(1,3)
hypercube3.add_edge(1,5)
hypercube3.add_edge(1,9)

hypercube3.add_edge(2,3)
hypercube3.add_edge(2,6)
hypercube3.add_edge(2,10)

hypercube3.add_edge(3,7)
hypercube3.add_edge(3,11)

hypercube3.add_edge(4,5)
hypercube3.add_edge(4,6)
hypercube3.add_edge(4,12)

hypercube3.add_edge(5,7)
hypercube3.add_edge(5,13)

hypercube3.add_edge(6,7)
hypercube3.add_edge(6,14)

hypercube3.add_edge(7,15)

hypercube3.add_edge(8,9)
hypercube3.add_edge(8,10)
hypercube3.add_edge(8,12)

hypercube3.add_edge(9,11)
hypercube3.add_edge(9,13)

hypercube3.add_edge(10,11)
hypercube3.add_edge(10,14)

hypercube3.add_edge(12,13)
hypercube3.add_edge(12,14)

hypercube3.add_edge(13,15)

hypercube3.add_edge(14,15)


tt = nx.MultiGraph()
tt.add_edge(0,1)
tt.add_edge(0,3)
tt.add_edge(1,2)
tt.add_edge(1,4)
tt.add_edge(2,5)
tt.add_edge(3,6)
tt.add_edge(4,7)
tt.add_edge(5,8)
tt.add_edge(6,7)
tt.add_edge(7,8)
tt.add_edge(8,9)
tt.add_edge(9,10)
tt.add_edge(10,11)


t2 = nx.MultiGraph()
t2.add_edge(0,1)
t2.add_edge(0,2)
t2.add_edge(1,2)
t2.add_edge(2,3)
t2.add_edge(3,4)
t2.add_edge(3,5)
t2.add_edge(4,5)


t3 = nx.MultiGraph()
t3.add_edge(0,1)
t3.add_edge(1,2)
t3.add_edge(1,3)
t3.add_edge(2,3)
t3.add_edge(3,4)
t3.add_edge(4,6)
t3.add_edge(4,5)
t3.add_edge(5,6)
t3.add_edge(6,7)
t3.add_edge(7,8)
t3.add_edge(7,9)
t3.add_edge(8,9)
t3.add_edge(9,10)
t3.add_edge(10,0)
t3.add_edge(10,11)
t3.add_edge(11,0)

c_breaker4 = nx.MultiGraph()
c_breaker4.add_edge(0,10)
c_breaker4.add_edge(8,9)
c_breaker4.add_edge(8,9)
c_breaker4.add_edge(8,5)
c_breaker4.add_edge(8,15)
c_breaker4.add_edge(2,3)
c_breaker4.add_edge(2,10)
c_breaker4.add_edge(1,3)
c_breaker4.add_edge(5,10)
c_breaker4.add_edge(5,10)
c_breaker4.add_edge(5,15)
c_breaker4.add_edge(3,11)
c_breaker4.add_edge(10,11)

c_breaker5 = nx.MultiGraph()
c_breaker5.add_edge(0,1)
c_breaker5.add_edge(0,2)
c_breaker5.add_edge(1,2)
c_breaker5.add_edge(2,3)
c_breaker5.add_edge(2,3)
c_breaker5.add_edge(2,4)
c_breaker5.add_edge(4,5)
c_breaker5.add_edge(4,6)
c_breaker5.add_edge(5,6)

c_breaker6 = nx.MultiGraph()
c_breaker6.add_edge(0,5)
c_breaker6.add_edge(8,10)
c_breaker6.add_edge(8,9)
c_breaker6.add_edge(2,6)
c_breaker6.add_edge(2,3)
c_breaker6.add_edge(1,3)
c_breaker6.add_edge(9,5)
c_breaker6.add_edge(5,12)
c_breaker6.add_edge(5,12) #Dup
c_breaker6.add_edge(5,6)
c_breaker6.add_edge(3,11)
c_breaker6.add_edge(10,11)


c_breaker7 = nx.MultiGraph()
c_breaker7.add_edge(0,4)
c_breaker7.add_edge(8,10)
c_breaker7.add_edge(8,9)
c_breaker7.add_edge(4,2)
c_breaker7.add_edge(4,5)
c_breaker7.add_edge(2,3)
c_breaker7.add_edge(1,3)
c_breaker7.add_edge(9,5)
c_breaker7.add_edge(5,15) #Dup
c_breaker7.add_edge(5,15)
c_breaker7.add_edge(3,11)
c_breaker7.add_edge(10,11)


mtree = nx.MultiGraph()
# Cycle
mtree.add_edge(0, 1)
mtree.add_edge(1, 2)
mtree.add_edge(1, 2)
mtree.add_edge(2, 0)
mtree.add_edge(2, 0)
mtree.add_edge(2, 0)
#Tree
mtree.add_edge(2, 3)
mtree.add_edge(2, 3)
mtree.add_edge(2, 3)
mtree.add_edge(2, 3)
mtree.add_edge(3, 4)
mtree.add_edge(3, 4)
#mtree.add_edge(4, 2)
#Cycle
mtree.add_edge(4, 5)
mtree.add_edge(4, 5)
mtree.add_edge(5, 6)
mtree.add_edge(5, 6)
mtree.add_edge(6, 7)
mtree.add_edge(6, 7)
mtree.add_edge(7, 4)
mtree.add_edge(7, 4)
#Tree
mtree.add_edge(7, 8)
mtree.add_edge(7, 8)
mtree.add_edge(7, 8)
mtree.add_edge(8, 9)
mtree.add_edge(9, 10)
mtree.add_edge(9, 10)

double_pentagons = nx.MultiGraph()
double_pentagons.add_edge(0,1)
double_pentagons.add_edge(0,2)
double_pentagons.add_edge(1,3)
double_pentagons.add_edge(3,5)
double_pentagons.add_edge(2,4)
double_pentagons.add_edge(4,5)
double_pentagons.add_edge(4,6)
double_pentagons.add_edge(6,8)
double_pentagons.add_edge(8,9)
double_pentagons.add_edge(9,7)
double_pentagons.add_edge(7,5)


# C_5 2 chords pentagon

c5_ch2_1 = nx.MultiGraph()
c5_ch2_1.add_edge(0, 1)
c5_ch2_1.add_edge(1, 2)
c5_ch2_1.add_edge(2, 3)
c5_ch2_1.add_edge(3, 4)
c5_ch2_1.add_edge(4, 0)
# Chords
c5_ch2_1.add_edge(0, 2)
c5_ch2_1.add_edge(0, 3)

c5_ch2_2 = nx.MultiGraph()
c5_ch2_2.add_edge(0, 1)
c5_ch2_2.add_edge(1, 2)
c5_ch2_2.add_edge(2, 3)
c5_ch2_2.add_edge(3, 4)
c5_ch2_2.add_edge(4, 0)
# Chords
c5_ch2_2.add_edge(0, 2)
c5_ch2_2.add_edge(1, 4)

# C_6 2 chords

c6_ch2_1 = nx.MultiGraph()
c6_ch2_1.add_edge(0, 1)
c6_ch2_1.add_edge(1, 2)
c6_ch2_1.add_edge(2, 3)
c6_ch2_1.add_edge(3, 4)
c6_ch2_1.add_edge(4, 5)
c6_ch2_1.add_edge(5, 0)
# Chords
c6_ch2_1.add_edge(0, 2)
c6_ch2_1.add_edge(0, 4)

c6_ch2_2 = nx.MultiGraph()
c6_ch2_2.add_edge(0, 1)
c6_ch2_2.add_edge(1, 2)
c6_ch2_2.add_edge(2, 3)
c6_ch2_2.add_edge(3, 4)
c6_ch2_2.add_edge(4, 5)
c6_ch2_2.add_edge(5, 0)
# Chords
c6_ch2_2.add_edge(1, 5)
c6_ch2_2.add_edge(2, 4)

c6_ch2_3 = nx.MultiGraph()
c6_ch2_3.add_edge(0, 1)
c6_ch2_3.add_edge(1, 2)
c6_ch2_3.add_edge(2, 3)
c6_ch2_3.add_edge(3, 4)
c6_ch2_3.add_edge(4, 5)
c6_ch2_3.add_edge(5, 0)
# Chords
c6_ch2_3.add_edge(0, 3)
c6_ch2_3.add_edge(1, 4)

c6_ch2_4 = nx.MultiGraph()
c6_ch2_4.add_edge(0, 1)
c6_ch2_4.add_edge(1, 2)
c6_ch2_4.add_edge(2, 3)
c6_ch2_4.add_edge(3, 4)
c6_ch2_4.add_edge(4, 5)
c6_ch2_4.add_edge(5, 0)
# Chords
c6_ch2_4.add_edge(0, 3)
c6_ch2_4.add_edge(0, 4)

c6_ch2_5 = nx.MultiGraph()
c6_ch2_5.add_edge(0, 1)
c6_ch2_5.add_edge(1, 2)
c6_ch2_5.add_edge(2, 3)
c6_ch2_5.add_edge(3, 4)
c6_ch2_5.add_edge(4, 5)
c6_ch2_5.add_edge(5, 0)
# Chords
c6_ch2_5.add_edge(1, 3)
c6_ch2_5.add_edge(2, 0)

c6_ch2_6 = nx.MultiGraph()
c6_ch2_6.add_edge(0, 1)
c6_ch2_6.add_edge(1, 2)
c6_ch2_6.add_edge(2, 3)
c6_ch2_6.add_edge(3, 4)
c6_ch2_6.add_edge(4, 5)
c6_ch2_6.add_edge(5, 0)
# Chords
c6_ch2_6.add_edge(0, 2)
c6_ch2_6.add_edge(1, 4)

# C_7 2 chords

c7_ch2_1 = nx.MultiGraph()
c7_ch2_1.add_edge(0, 1)
c7_ch2_1.add_edge(1, 2)
c7_ch2_1.add_edge(2, 3)
c7_ch2_1.add_edge(3, 4)
c7_ch2_1.add_edge(4, 5)
c7_ch2_1.add_edge(5, 6)
c7_ch2_1.add_edge(6, 0)
# Chords
c7_ch2_1.add_edge(0, 4)
c7_ch2_1.add_edge(6, 2)

c7_ch2_2 = nx.MultiGraph()
c7_ch2_2.add_edge(0, 1)
c7_ch2_2.add_edge(1, 2)
c7_ch2_2.add_edge(2, 3)
c7_ch2_2.add_edge(3, 4)
c7_ch2_2.add_edge(4, 5)
c7_ch2_2.add_edge(5, 6)
c7_ch2_2.add_edge(6, 0)
# Chords
c7_ch2_2.add_edge(0, 5)
c7_ch2_2.add_edge(2, 4)

c7_ch2_3 = nx.MultiGraph()
c7_ch2_3.add_edge(0, 1)
c7_ch2_3.add_edge(1, 2)
c7_ch2_3.add_edge(2, 3)
c7_ch2_3.add_edge(3, 4)
c7_ch2_3.add_edge(4, 5)
c7_ch2_3.add_edge(5, 6)
c7_ch2_3.add_edge(6, 0)
# Chords
c7_ch2_3.add_edge(0, 4)
c7_ch2_3.add_edge(1, 3)

c7_ch2_4 = nx.MultiGraph()
c7_ch2_4.add_edge(0, 1)
c7_ch2_4.add_edge(1, 2)
c7_ch2_4.add_edge(2, 3)
c7_ch2_4.add_edge(3, 4)
c7_ch2_4.add_edge(4, 5)
c7_ch2_4.add_edge(5, 6)
c7_ch2_4.add_edge(6, 0)
# Chords
c7_ch2_4.add_edge(0, 4)
c7_ch2_4.add_edge(0, 3)

c7_ch2_5 = nx.MultiGraph()
c7_ch2_5.add_edge(0, 1)
c7_ch2_5.add_edge(1, 2)
c7_ch2_5.add_edge(2, 3)
c7_ch2_5.add_edge(3, 4)
c7_ch2_5.add_edge(4, 5)
c7_ch2_5.add_edge(5, 6)
c7_ch2_5.add_edge(6, 0)
# Chords
c7_ch2_5.add_edge(0, 5)
c7_ch2_5.add_edge(0, 2)

c7_ch2_6 = nx.MultiGraph()
c7_ch2_6.add_edge(0, 1)
c7_ch2_6.add_edge(1, 2)
c7_ch2_6.add_edge(2, 3)
c7_ch2_6.add_edge(3, 4)
c7_ch2_6.add_edge(4, 5)
c7_ch2_6.add_edge(5, 6)
c7_ch2_6.add_edge(6, 0)
# Chords
c7_ch2_6.add_edge(0, 4)
c7_ch2_6.add_edge(0, 5)

c7_ch2_7 = nx.MultiGraph()
c7_ch2_7.add_edge(0, 1)
c7_ch2_7.add_edge(1, 2)
c7_ch2_7.add_edge(2, 3)
c7_ch2_7.add_edge(3, 4)
c7_ch2_7.add_edge(4, 5)
c7_ch2_7.add_edge(5, 6)
c7_ch2_7.add_edge(6, 0)
# Chords
c7_ch2_7.add_edge(0, 5)
c7_ch2_7.add_edge(6, 1)

c7_ch2_8 = nx.MultiGraph()
c7_ch2_8.add_edge(0, 1)
c7_ch2_8.add_edge(1, 2)
c7_ch2_8.add_edge(2, 3)
c7_ch2_8.add_edge(3, 4)
c7_ch2_8.add_edge(4, 5)
c7_ch2_8.add_edge(5, 6)
c7_ch2_8.add_edge(6, 0)
# Chords
c7_ch2_8.add_edge(0, 4)
c7_ch2_8.add_edge(3, 5)

c7_ch2_9 = nx.MultiGraph()
c7_ch2_9.add_edge(0, 1)
c7_ch2_9.add_edge(1, 2)
c7_ch2_9.add_edge(2, 3)
c7_ch2_9.add_edge(3, 4)
c7_ch2_9.add_edge(4, 5)
c7_ch2_9.add_edge(5, 6)
c7_ch2_9.add_edge(6, 0)
# Chords
c7_ch2_9.add_edge(1, 5)
c7_ch2_9.add_edge(2, 6)

# C_6 3 chords

c6_ch3_1 = nx.MultiGraph()
c6_ch3_1.add_edge(0, 1)
c6_ch3_1.add_edge(1, 2)
c6_ch3_1.add_edge(2, 3)
c6_ch3_1.add_edge(3, 4)
c6_ch3_1.add_edge(4, 5)
c6_ch3_1.add_edge(5, 0)
# Chords
c6_ch3_1.add_edge(0, 2)
c6_ch3_1.add_edge(0, 3)
c6_ch3_1.add_edge(0, 4)

c6_ch3_2 = nx.MultiGraph()
c6_ch3_2.add_edge(0, 1)
c6_ch3_2.add_edge(1, 2)
c6_ch3_2.add_edge(2, 3)
c6_ch3_2.add_edge(3, 4)
c6_ch3_2.add_edge(4, 5)
c6_ch3_2.add_edge(5, 0)
# Chords
c6_ch3_2.add_edge(0, 4)
c6_ch3_2.add_edge(0, 3)
c6_ch3_2.add_edge(2, 5)

c6_ch3_3 = nx.MultiGraph()
c6_ch3_3.add_edge(0, 1)
c6_ch3_3.add_edge(1, 2)
c6_ch3_3.add_edge(2, 3)
c6_ch3_3.add_edge(3, 4)
c6_ch3_3.add_edge(4, 5)
c6_ch3_3.add_edge(5, 0)
# Chords
c6_ch3_3.add_edge(0, 3)
c6_ch3_3.add_edge(0, 4)
c6_ch3_3.add_edge(1, 5)

c6_ch3_4 = nx.MultiGraph()
c6_ch3_4.add_edge(0, 1)
c6_ch3_4.add_edge(1, 2)
c6_ch3_4.add_edge(2, 3)
c6_ch3_4.add_edge(3, 4)
c6_ch3_4.add_edge(4, 5)
c6_ch3_4.add_edge(5, 0)
# Chords
c6_ch3_4.add_edge(1, 4)
c6_ch3_4.add_edge(0, 2)
c6_ch3_4.add_edge(3, 5)

c6_ch3_5 = nx.MultiGraph()
c6_ch3_5.add_edge(0, 1)
c6_ch3_5.add_edge(1, 2)
c6_ch3_5.add_edge(2, 3)
c6_ch3_5.add_edge(3, 4)
c6_ch3_5.add_edge(4, 5)
c6_ch3_5.add_edge(5, 0)
# Chords
c6_ch3_5.add_edge(1, 4)
c6_ch3_5.add_edge(2, 5)
c6_ch3_5.add_edge(0, 3)

c6_ch3_6 = nx.MultiGraph()
c6_ch3_6.add_edge(0, 1)
c6_ch3_6.add_edge(1, 2)
c6_ch3_6.add_edge(2, 3)
c6_ch3_6.add_edge(3, 4)
c6_ch3_6.add_edge(4, 5)
c6_ch3_6.add_edge(5, 0)
# Chords
c6_ch3_6.add_edge(0, 2)
c6_ch3_6.add_edge(0, 4)
c6_ch3_6.add_edge(1, 3)

c6_ch3_7 = nx.MultiGraph()
c6_ch3_7.add_edge(0, 1)
c6_ch3_7.add_edge(1, 2)
c6_ch3_7.add_edge(2, 3)
c6_ch3_7.add_edge(3, 4)
c6_ch3_7.add_edge(4, 5)
c6_ch3_7.add_edge(5, 0)
# Chords
c6_ch3_7.add_edge(0, 2)
c6_ch3_7.add_edge(0, 4)
c6_ch3_7.add_edge(2, 4)

# C_6 5 chords

c6_ch5_1 = nx.MultiGraph()
c6_ch5_1.add_edge(0, 1)
c6_ch5_1.add_edge(1, 2)
c6_ch5_1.add_edge(2, 3)
c6_ch5_1.add_edge(3, 4)
c6_ch5_1.add_edge(4, 5)
c6_ch5_1.add_edge(5, 0)
# Chords
c6_ch5_1.add_edge(0, 3)
c6_ch5_1.add_edge(1, 4)
c6_ch5_1.add_edge(2, 5)

c6_ch5_1.add_edge(0, 4)

# C_7 3 chords

c7_ch3_1 = nx.MultiGraph()
c7_ch3_1.add_edge(0, 1)
c7_ch3_1.add_edge(1, 2)
c7_ch3_1.add_edge(2, 3)
c7_ch3_1.add_edge(3, 4)
c7_ch3_1.add_edge(4, 5)
c7_ch3_1.add_edge(5, 6)
c7_ch3_1.add_edge(6, 0)
# Chords
c7_ch3_1.add_edge(0, 3)
c7_ch3_1.add_edge(1, 5)
c7_ch3_1.add_edge(4, 6)

c7_ch3_2 = nx.MultiGraph()
c7_ch3_2.add_edge(0, 1)
c7_ch3_2.add_edge(1, 2)
c7_ch3_2.add_edge(2, 3)
c7_ch3_2.add_edge(3, 4)
c7_ch3_2.add_edge(4, 5)
c7_ch3_2.add_edge(5, 6)
c7_ch3_2.add_edge(6, 0)
# Chords
c7_ch3_2.add_edge(0, 3)
c7_ch3_2.add_edge(1, 5)
c7_ch3_2.add_edge(2, 6)

c7_ch3_3 = nx.MultiGraph()
c7_ch3_3.add_edge(0, 1)
c7_ch3_3.add_edge(1, 2)
c7_ch3_3.add_edge(2, 3)
c7_ch3_3.add_edge(3, 4)
c7_ch3_3.add_edge(4, 5)
c7_ch3_3.add_edge(5, 6)
c7_ch3_3.add_edge(6, 0)
# Chords
c7_ch3_3.add_edge(0, 3)
c7_ch3_3.add_edge(1, 5)
c7_ch3_3.add_edge(2, 5)

# C_7 4 chords

c7_ch4_1 = nx.MultiGraph()
c7_ch4_1.add_edge(0, 1)
c7_ch4_1.add_edge(1, 2)
c7_ch4_1.add_edge(2, 3)
c7_ch4_1.add_edge(3, 4)
c7_ch4_1.add_edge(4, 5)
c7_ch4_1.add_edge(5, 6)
c7_ch4_1.add_edge(6, 0)
# Chords
c7_ch4_1.add_edge(0, 3)
c7_ch4_1.add_edge(0, 4)
c7_ch4_1.add_edge(1, 5)
c7_ch4_1.add_edge(2, 6)

# C_10 2 chords

c10_ch2_1 = nx.MultiGraph()
c10_ch2_1.add_edge(0, 1)
c10_ch2_1.add_edge(1, 2)
c10_ch2_1.add_edge(2, 3)
c10_ch2_1.add_edge(3, 4)
c10_ch2_1.add_edge(4, 5)
c10_ch2_1.add_edge(5, 6)
c10_ch2_1.add_edge(6, 7)
c10_ch2_1.add_edge(7, 8)
c10_ch2_1.add_edge(8, 9)
c10_ch2_1.add_edge(9, 0)
# Chords
c10_ch2_1.add_edge(0, 5)
c10_ch2_1.add_edge(2, 7)

c10_ch2_2 = nx.MultiGraph()
c10_ch2_2.add_edge(0, 1)
c10_ch2_2.add_edge(1, 2)
c10_ch2_2.add_edge(2, 3)
c10_ch2_2.add_edge(3, 4)
c10_ch2_2.add_edge(4, 5)
c10_ch2_2.add_edge(5, 6)
c10_ch2_2.add_edge(6, 7)
c10_ch2_2.add_edge(7, 8)
c10_ch2_2.add_edge(8, 9)
c10_ch2_2.add_edge(9, 0)
# Chords
c10_ch2_2.add_edge(0, 5)
c10_ch2_2.add_edge(2, 7)
c10_ch2_2.add_edge(1, 6)

# C_8 7 chords

c8_ch7_1 = nx.MultiGraph()
c8_ch7_1.add_edge(0, 1)
c8_ch7_1.add_edge(1, 2)
c8_ch7_1.add_edge(2, 3)
c8_ch7_1.add_edge(3, 4)
c8_ch7_1.add_edge(4, 5)
c8_ch7_1.add_edge(5, 6)
c8_ch7_1.add_edge(6, 7)
c8_ch7_1.add_edge(7, 0)
# Chords
c8_ch7_1.add_edge(0, 4)
c8_ch7_1.add_edge(1, 5)
c8_ch7_1.add_edge(2, 6)
c8_ch7_1.add_edge(3, 7)

c8_ch7_1.add_edge(0, 5)
c8_ch7_1.add_edge(1, 4)
c8_ch7_1.add_edge(2, 7)

c8_ch7_2 = nx.MultiGraph()
c8_ch7_2.add_edge(0, 1)
c8_ch7_2.add_edge(1, 2)
c8_ch7_2.add_edge(2, 3)
c8_ch7_2.add_edge(3, 4)
c8_ch7_2.add_edge(4, 5)
c8_ch7_2.add_edge(5, 6)
c8_ch7_2.add_edge(6, 7)
c8_ch7_2.add_edge(7, 0)
# Chords
c8_ch7_2.add_edge(0, 4)
c8_ch7_2.add_edge(1, 5)
c8_ch7_2.add_edge(2, 6)
c8_ch7_2.add_edge(3, 7)

c8_ch7_2.add_edge(0, 5)
c8_ch7_2.add_edge(1, 4)
c8_ch7_1.add_edge(2, 5)

# C_8 3 chords

c8_ch3_1 = nx.MultiGraph()
c8_ch3_1.add_edge(0, 1)
c8_ch3_1.add_edge(1, 2)
c8_ch3_1.add_edge(2, 3)
c8_ch3_1.add_edge(3, 4)
c8_ch3_1.add_edge(4, 5)
c8_ch3_1.add_edge(5, 6)
c8_ch3_1.add_edge(6, 7)
c8_ch3_1.add_edge(7, 0)
# Chords
c8_ch3_1.add_edge(0, 4)
c8_ch3_1.add_edge(1, 5)
c8_ch3_1.add_edge(3, 7)

c8_ch3_2 = nx.MultiGraph()
c8_ch3_2.add_edge(0, 1)
c8_ch3_2.add_edge(1, 2)
c8_ch3_2.add_edge(2, 3)
c8_ch3_2.add_edge(3, 4)
c8_ch3_2.add_edge(4, 5)
c8_ch3_2.add_edge(5, 6)
c8_ch3_2.add_edge(6, 7)
c8_ch3_2.add_edge(7, 0)
# Chords
c8_ch3_2.add_edge(0, 4)
c8_ch3_2.add_edge(2, 5)
c8_ch3_2.add_edge(3, 7)
"""
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

hams = GraphTools.gen_all_hamiltonian(10, 20)

Benchmarks.relpoly_binary_improved_console_benchmark(hams, 100)