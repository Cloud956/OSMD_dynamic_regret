from Graphs import *
from Algorithms import *

# generate a gridgraph where the weights for edges are pulled from the below defined distribution
distribution = lambda: round(np.random.normal(20, 5), 2)     # select the distribution to be used
gg = GridGraph(3, 3, distribution)

algs = Algorithms(gg) # initialize algorithms class for the generated graph

s = 0   # start vector
t = 2   # destination vector
m = 6   # path length

# generate and print all paths of length
paths = algs.bfs_paths(s, t, m)
print("Vertex paths:\n" + str(paths))

# generate and print the binary encoded paths
encoded_paths = algs.encode(paths)
print("\nEncoded paths:\n" + str(encoded_paths))

# run the EXP(2) algorithm
algs.exp2(0.01, paths=encoded_paths, rounds=10000)

# visualize the graph
gv = GridGraphVisualizer(gg) # for the visualization, each edge has a tuple for the label with the following structure: (id, weight)
gv.visualize(s, t)