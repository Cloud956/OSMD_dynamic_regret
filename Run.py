from Graphs import *
from Algorithms import *
import matplotlib.pyplot as plt

np.random.seed(0)

# generate a gridgraph where the weights for edges are pulled from the below defined distribution
distribution = lambda: round(np.random.normal(100, 50), 0)     # select the distribution to be used
gg = GridGraph(3, 3, distribution)

algs = Algorithms(gg) # initialize algorithms class for the generated graph

s = 0   # start vector
t = 2   # destination vector
m = 6   # path length

# generate and print all paths of length m
paths = algs.bfs_paths(s, t, m)
print("\nVertex paths:\n" + str(paths))

# generate and print the binary encoded paths
encoded_paths = algs.encode(paths)
print("\nEncoded paths:\n" + str(encoded_paths) + "\n")

# run the EXP2 algorithm
total_regrets = algs.exp2(eta=0.01, actions=encoded_paths, rounds=100000, game='full', debug=False) # set game = 'full' to usse exp2 with full information

# plot total regret over time
indices = range(len(total_regrets))
plt.plot(indices, total_regrets)
plt.xlabel('Round')
plt.ylabel('Total regret')
plt.title('Total regret for each round')
plt.show()

# visualize the graph
gv = GridGraphVisualizer(gg) # for the visualization, each edge has a tuple for the label with the following structure: (id, weight)
gv.visualize(s, t)