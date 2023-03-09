from Graphs import *
import numpy as np

class Algorithms:

    def __init__(self, graph):
        self.graph = graph

    def bfs_paths(self, s, t, m):
        paths = []
        queue = [(s, [s])]
        while queue:
            (vertex, path) = queue.pop(0)
            visited = set(path)
            for neighbor in self.graph.get_neighbors(vertex):
                if neighbor not in visited and len(path) <= m:
                    if neighbor == t and len(path) == m:
                        paths.append(path + [neighbor])
                    else:
                        queue.append((neighbor, path + [neighbor]))
        return paths
    
    def encode(self, paths):
        print("\nEdge paths:")
        encoded_paths = []
        for path in paths:
            edges = []
            for i in range(len(path)-1):
                edges.append(self.graph.get_edge_id(path[i], path[i+1]))

            print(edges)
            encoded_paths.append(self.get_binary_list(edges))

        return encoded_paths

    def get_binary_list(self, input_list):
        # Create an empty binary list of the desired size
        binary_list = [0] * self.graph.num_edges()

        # For each integer in the input list, set the corresponding index in the binary list to 1
        for num in input_list:
            binary_list[num] = 1

        return binary_list

    def action(self, P):
        U = np.random.uniform()
        CDF = [sum(P[:i+1]) for i in range(len(P))]
        i = 0
        while CDF[i] < U:
            i += 1
        return i
    
    def update_probabilities(self, probabilities: list, costs: list, eta: float, paths: list):
        P_new = [None] * len(paths)
        for a in range(len(paths)):
            numerator = probabilities[a] * np.exp(-eta * np.inner(costs, paths[a]))
            denominator = 0
            for b in range(len(paths)):
                denominator += probabilities[b] * np.exp(-eta * np.inner(costs, paths[b]))

            P_new[a] = numerator / denominator
        return P_new

    def minimal_loss(self, costs: list, paths: list):
        min = float('inf')
        for path in paths:
            loss = self.loss(costs, path)
            if loss < min:
                min = loss
        return min

    def loss(self, costs: list, path: list):
        return np.inner(costs, path)

    def exp2(self, eta: float, paths: list, rounds: int):
        # initialize probability vector
        probabilities = [1/len(paths)] * len(paths)
        regret = []

        # loop through the rounds
        for t in range(1, rounds+1):
            
            # check if probabilities have converged
            for i in probabilities:
                if (1-i) <= 0.001:
                    return regret

            print("\nround " + str(t) + " ↓")
            self.graph.update_edge_weights()                            # "adversary" generates new edge weights

            print("probabilities =\t\t" + str(np.round(probabilities, 2)))

            # choose an action
            action = self.action(probabilities)                         # decision maker chooses action based on probability vector P_t
            print("action =\t\t" + str(action))

            # reveal costs
            costs = self.graph.get_all_edge_weights()                   # decision maker is "informed" of loss vector z_t
            print("costs =\t\t\t" + str(costs))

            # calculate regret
            actual_loss = self.loss(costs, paths[action])
            minimum_loss = self.minimal_loss(costs, paths)
            regret.append(actual_loss - minimum_loss)
            print("regret =\t\t" + str(regret[-1]))

            # calculate P_t+1
            probabilities = self.update_probabilities(probabilities, costs, eta, paths)
                
        return regret
