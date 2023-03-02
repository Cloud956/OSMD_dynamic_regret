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


    def exp2(self, eta: float, paths: list, rounds: int):
        # initialize probability vector
        P = [1/len(paths)] * len(paths)
        print("P_1 = " + str(P))

        # loop through the rounds
        for t in range(1, rounds+1):
            a = self.action(P)
            print("a_" + str(t) + " = " + str(a))

            z = self.graph.get_all_edge_weights()
            print("z_" + str(t) + " = " + str(z))

            P_new = [None] * len(paths)
            for a in range(len(paths)):
                numerator = P[a] * np.exp(-eta * np.inner(z, paths[a]))
                denominator = 0
                for b in range(len(paths)):
                    if b != a:
                        denominator += P[b] * np.exp(-eta * np.inner(z, paths[b]))

                P_new[a] = numerator / denominator

            P_new = P_new/np.sum(P_new)
            P = P_new
            print("P_" + str(t) + " = " + str(np.round(P, 2)))
            self.graph.update_edge_weights()

            # check if probabilities have converged
            for i in P:
                if (1-i) <= 0.0001:
                    return
