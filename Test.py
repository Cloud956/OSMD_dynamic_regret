from Graphs import *
from Algorithms import *

""" g = Graph()

g.add_edge(0, 1, 1)
g.add_edge(1, 2, 1)
g.add_edge(2, 0, 1) """

gg = GridGraph(3, 3, lambda: round(np.random.normal(2, 1), 2))

algs = Algorithms(gg)

s = 0 # start vector
t = 2 # destination vector

paths = algs.bfs_paths(s, t, 6)
print("Vertex paths:\n" + str(paths))

encoded_paths = algs.encode(paths)
print("\nEncoded paths:\n" + str(encoded_paths))

algs.exp2(0.1, encoded_paths, 15)

gv = GridGraphVisualizer(gg)
gv.visualize(s, t)