from Graphs import *
from Algorithms import *
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
#np.random.seed(3)

# generate a gridgraph where the weights for edges are pulled from the below defined distribution
distribution = lambda: round(np.random.uniform(0, 10), 0)     # select the distribution to be used
gg = GridGraph(3, 3, distribution)

algs = Algorithms(gg) # initialize algorithms class for the generated graph

s = 0   # start vector
t = 8   # destination vector
m = 6   # path length

# generate and print all paths of length m
paths = algs.bfs_paths(s, t, m)
print("\nVertex paths:\n" + str(paths))

# generate and print the binary encoded paths
encoded_paths = algs.encode(paths)
print("\nEncoded paths:\n" + str(encoded_paths) + "\n")

""" # run the EXP2 algorithm
total_regrets_full = algs.exp2(eta=eta, actions=encoded_paths, rounds=nrounds, game='full', debug=False) # set game = 'full' to use exp2 with full information
total_regrets_semi = algs.exp2(eta=eta, actions=encoded_paths, rounds=nrounds, game='semi', debug=False) # set game = 'full' to use exp2 with full information
total_regrets_bandit = algs.exp2(eta=eta, actions=encoded_paths, rounds=nrounds, game='bandit', debug=False) # set game = 'full' to use exp2 with full information

total_regrets_osmd_full = algs.osmd(eta=0.01, actions=encoded_paths, rounds=nrounds, game='bandit', debug=False)
total_regrets_osmd_semi = algs.osmd(eta=0.01, actions=encoded_paths, rounds=nrounds, game='bandit', debug=False)
total_regrets_osmd_bandit = algs.osmd(eta=0.01, actions=encoded_paths, rounds=nrounds, game='bandit', debug=False)

# plot total regret over time for EXP2 algorithm
plt.figure('exp2')
indices = range(len(total_regrets_full))
plt.plot(indices, total_regrets_full, label='exp2 full')
plt.xlabel('Round')
plt.ylabel('Total regret')
plt.title('Total regret for each round')

indices = range(len(total_regrets_semi))
plt.plot(indices, total_regrets_semi, label='exp2 semi')
plt.xlabel('Round')
plt.ylabel('Total regret')
plt.title('Total regret for each round')

indices = range(len(total_regrets_bandit))
plt.plot(indices, total_regrets_bandit, label='exp2 bandit')
plt.xlabel('Round')
plt.ylabel('Total regret')
plt.title('Total regret for each round')
plt.legend()

# plot total regret over time for OSMD algorithm
plt.figure('osmd')
indices = range(len(total_regrets_osmd_full))
plt.plot(indices, total_regrets_osmd_full, label='osmd full')
plt.xlabel('Round')
plt.ylabel('Total regret')
plt.title('Total regret for each round')

indices = range(len(total_regrets_osmd_semi))
plt.plot(indices, total_regrets_osmd_semi, label='osmd semi')
plt.xlabel('Round')
plt.ylabel('Total regret')
plt.title('Total regret for each round')

indices = range(len(total_regrets_osmd_bandit))
plt.plot(indices, total_regrets_osmd_bandit, label='osmd bandit')
plt.xlabel('Round')
plt.ylabel('Total regret')
plt.title('Total regret for each round')
plt.legend()

# visualize the graph
gv = GridGraphVisualizer(gg) # for the visualization, each edge has a tuple for the label with the following structure: (id, weight)
gv.visualize(s, t)
plt.show() """

nrounds = 1000
eta = 0.05

total_regrets_full_sum = np.array([0] * nrounds)
total_regrets_semi_sum = np.array([0] * nrounds)
total_regrets_bandit_sum = np.array([0] * nrounds)

total_regrets_osmd_full_sum = np.array([0] * nrounds)
total_regrets_osmd_semi_sum = np.array([0] * nrounds)
total_regrets_osmd_bandit_sum = np.array([0] * nrounds)

nruns = 1 #1000

progress = 0

for i in range(1,nruns+1):
    print(i)

    try:
        total_regrets_full = algs.exp2(eta=eta, actions=encoded_paths, rounds=nrounds, game='full') # set game = 'full' to use exp2 with full information
        total_regrets_semi = algs.exp2(eta=eta, actions=encoded_paths, rounds=nrounds, game='semi') # set game = 'full' to use exp2 with full information
        total_regrets_bandit = algs.exp2(eta=eta, actions=encoded_paths, rounds=nrounds, game='bandit') # set game = 'full' to use exp2 with full information

        total_regrets_osmd_full = algs.osmd(eta=eta, actions=encoded_paths, rounds=nrounds, game='full') # they were all bandit :(
        total_regrets_osmd_semi = algs.osmd(eta=eta, actions=encoded_paths, rounds=nrounds, game='semi')
        total_regrets_osmd_bandit = algs.osmd(eta=eta, actions=encoded_paths, rounds=nrounds, game='bandit')
    except:
           nruns -=1
           print(nruns)
           continue

    total_regrets_full_sum = np.add(total_regrets_full_sum, total_regrets_full)
    total_regrets_semi_sum = np.add(total_regrets_semi_sum, total_regrets_semi)
    total_regrets_bandit_sum = np.add(total_regrets_bandit_sum, total_regrets_bandit)

    total_regrets_osmd_full_sum = np.add(total_regrets_osmd_full_sum, total_regrets_osmd_full)
    total_regrets_osmd_semi_sum = np.add(total_regrets_osmd_semi_sum, total_regrets_osmd_semi)
    total_regrets_osmd_bandit_sum = np.add(total_regrets_osmd_bandit_sum, total_regrets_osmd_bandit)

print('done:', nruns)

total_regrets_full = total_regrets_full_sum / nruns
total_regrets_semi = total_regrets_semi_sum / nruns
total_regrets_bandit = total_regrets_bandit_sum / nruns

total_regrets_osmd_full = total_regrets_osmd_full_sum / nruns
total_regrets_osmd_semi = total_regrets_osmd_semi_sum / nruns
total_regrets_osmd_bandit = total_regrets_osmd_bandit_sum / nruns

# plot total regret over time for EXP2 algorithm
plt.figure('exp2')
indices = range(len(total_regrets_full))
plt.plot(indices, total_regrets_full, linewidth=1, label='exp2 full')
plt.xlabel('Round')
plt.ylabel('Average regret')
plt.title('Average regret at each round for exp2 (' + str(nruns) +' runs)')

indices = range(len(total_regrets_semi))
plt.plot(indices, total_regrets_semi, linewidth=1, label='exp2 semi')

indices = range(len(total_regrets_bandit))
plt.plot(indices, total_regrets_bandit, linewidth=1, label='exp2 bandit')
plt.legend()

plt.savefig('exp2.png', dpi=300)

# plot total regret over time for OSMD algorithm
plt.figure('osmd')
indices = range(len(total_regrets_osmd_full))
plt.plot(indices, total_regrets_osmd_full, linewidth=1, label='osmd full')
plt.xlabel('Round')
plt.ylabel('Average regret')
plt.title('Average regret at each round for osmd (' + str(nruns) +' runs)')

indices = range(len(total_regrets_osmd_semi))
plt.plot(indices, total_regrets_osmd_semi, linewidth=1, label='osmd semi')

indices = range(len(total_regrets_osmd_bandit))
plt.plot(indices, total_regrets_osmd_bandit, linewidth=1, label='osmd bandit')
plt.legend()

plt.savefig('osmd.png', dpi=300)

# visualize the graph
gv = GridGraphVisualizer(gg) # for the visualization, each edge has a tuple for the label with the following structure: (id, weight)
gv.visualize(s, t)
plt.show()