from Graphs import *
import numpy as np

class Algorithms:

    def __init__(self, graph: None):
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
    
    def update_probabilities(self, probabilities: list, loss_vector: list, eta: float, paths: list):
        P_new = [None] * len(paths)
        denominator = 0
        
        for b in range(len(paths)):
                denominator += probabilities[b] * np.exp(-eta * np.inner(loss_vector, paths[b]))

        for a in range(len(paths)):
            numerator = probabilities[a] * np.exp(-eta * np.inner(loss_vector, paths[a]))
            P_new[a] = numerator / denominator
        
        return P_new

    def min_total_loss(self, costs: list, paths: list):
        min = float('inf')
        action = -1
        for i in range(len(paths)):
            total_loss = 0
            for cost in costs:
                total_loss += self.loss(cost, paths[i])
            if total_loss < min:
                min = total_loss
                action = i
        return action, min

    def loss(self, cost: list, path: list):
        return np.inner(cost, path)
    
    def cov_matrix(self, probabilities: list, actions: list):
        cov_matrix = 0
        for a in range(len(actions)):
            square_matrix = np.outer(actions[a], actions[a])
            cov_matrix += probabilities[a] * square_matrix
        return cov_matrix

    def estimate_loss(self, probabilities, actions):
        return

    def exp2(self, eta: float, paths: list, rounds: int, game='full'):
        print("running exp2 algo (setting: " + game + ")...")

        # initialize probability vector
        probabilities = [1/len(paths)] * len(paths)
        losses = []
        loss_vectors = []
        progress = 0

        # loop through the rounds
        for t in range(1, rounds+1):
            if t % (rounds/10) == 0:
                progress += 10
                print('round progress: ' + str(progress) + '%')

            #print("\nround " + str(t) + " ↓")
            self.graph.update_edge_weights()                            # "adversary" generates new edge weights

            #print("probabilities =\t\t" + str(np.round(probabilities, 2)))

            # choose an action
            action = self.action(probabilities)                         # decision maker chooses action based on probability vector P_t
            #print("action =\t\t" + str(action))

            loss_vector = self.graph.get_all_edge_weights()
            loss_vectors.append(loss_vector) 

            instantaneous_loss = self.loss(loss_vector, paths[action])
            losses.append(instantaneous_loss)

            # set cost
            if game == 'bandit':
                del loss_vector
                pinv_cov_matrix = np.linalg.pinv(self.cov_matrix(probabilities, paths))
                estimated_loss_vector = pinv_cov_matrix @ paths[action] * instantaneous_loss
                loss_vector = estimated_loss_vector

            # calculate P_t+1
            probabilities = self.update_probabilities(probabilities, loss_vector, eta, paths)

        print("\ncalculating total regret...")

        expected_loss = sum(losses) / len(losses)
        best_action, total_loss_best_action = self.min_total_loss(loss_vectors, paths)
        expected_loss_best_action = total_loss_best_action / len(loss_vectors)
        
        print("\nfinal probabilities =\t\t\t\t" + str(np.round(probabilities, 2)))

        print("best action (in hindsight) =\t\t\t" + str(best_action))

        print("\nexpected loss =\t\t\t\t\t" + str(expected_loss))
        print("expected loss for best action overall =\t\t" + str(expected_loss_best_action))

        total_regret = expected_loss - expected_loss_best_action

        return total_regret
