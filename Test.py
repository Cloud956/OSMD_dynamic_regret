from Graphs import *
from Algorithms import *

g = Graph()

g.add_edge(0, 1, 5)
g.add_edge(0, 2, 3)
g.add_edge(1, 2, 1)
g.add_edge(1, 3, 4)
g.add_edge(2, 3, 2)
g.add_edge(2, 1, 5)
g.add_edge(3, 2, 3)
g.add_edge(5, 2, 1)
g.add_edge(6, 3, 4)
g.add_edge(3, 5, 2)

gg = GridGraph(4)

algs = Algorithms(gg)

paths = algs.bfs_paths(0, 8, 4)
print("Vertex paths:", paths)

encoded_paths = algs.encode(paths)
print("Encoded paths:", encoded_paths)


gv = GridGraphVisualizer(gg)
gv.visualize()