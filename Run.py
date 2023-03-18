from Graphs import *
from Algorithms import *

#np.random.seed(0)

# generate a gridgraph where the weights for edges are pulled from the below defined distribution
distribution = lambda: round(np.random.normal(10, 3), 1)     # select the distribution to be used
gg = GridGraph(3, 3, distribution)

algs = Algorithms(gg) # initialize algorithms class for the generated graph

s = 0   # start vector
t = 2   # destination vector
m = 6   # path length

# generate and print all paths of length m
paths = algs.bfs_paths(s, t, m)
print("Vertex paths:\n" + str(paths))

# generate and print the binary encoded paths
encoded_paths = algs.encode(paths)
print("\nEncoded paths:\n" + str(encoded_paths) + "\n")

# run the EXP2 algorithm
total_regret = algs.exp2(eta=0.01, actions=encoded_paths, rounds=100000, game='bandit', debug=False) # set game = 'full' to usse exp2 with full information

# Calculate and print the regret over all the rounds
print("\ntotal regret =\t\t\t\t\t" + str(total_regret))

# visualize the graph
gv = GridGraphVisualizer(gg) # for the visualization, each edge has a tuple for the label with the following structure: (id, weight)
gv.visualize(s, t)