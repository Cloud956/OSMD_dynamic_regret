from Graphs import *
import numpy as np

class Algorithms:

    def __init__(self, graph: Graph):
        self.graph = graph

    def bfs_paths(self, s, t, m):
        '''
        computes and returns all vertex paths from vertex s to t with length = m
        '''
        actions = []
        queue = [(s, [s])]
        while queue:
            (vertex, path) = queue.pop(0)
            visited = set(path)
            for neighbor in self.graph.get_neighbors(vertex):
                if neighbor not in visited and len(path) <= m:
                    if neighbor == t and len(path) == m:
                        actions.append(path + [neighbor])
                    else:
                        queue.append((neighbor, path + [neighbor]))
        return actions
    
    def encode(self, actions):
        '''
        turns vector actions into edge actions and returns a binary encoded version
        '''
        print("\nEdge actions:")
        encoded_actions = []
        for path in actions:
            edges = []
            for i in range(len(path)-1):
                edges.append(self.graph.get_edge_id(path[i], path[i+1]))

            print(edges)
            encoded_actions.append(self.binary_list(edges))

        return encoded_actions

    def binary_list(self, input_list):
        '''
        turns an edge path into a binary encoding
        '''
        # Create an empty binary list of the desired size
        binary_list = [0] * self.graph.num_edges()

        # For each integer in the input list, set the corresponding index in the binary list to 1
        for num in input_list:
            binary_list[num] = 1

        return binary_list

    def action(self, prob_vector):
        '''
        creates a CDF from the provided probability vector, draws an action from that and returns the index
        '''
        U = np.random.uniform()
        CDF = [sum(prob_vector[:i+1]) for i in range(len(prob_vector))]
        i = 0
        while CDF[i] < U:
            i += 1
        return i
    
    def update_prob_vector(self, prob_vector: list, loss_vector: list, eta: float, actions: list):
        '''
        updates the probability vector p_t based on the (estimated) loss vector
        '''
        prob_vector_new = [None] * len(actions)
        denominator = 0
        
        for b in range(len(actions)):
                denominator += prob_vector[b] * np.exp(-eta * np.inner(loss_vector, actions[b]))

        for a in range(len(actions)):
            numerator = prob_vector[a] * np.exp(-eta * np.inner(loss_vector, actions[a]))
            prob_vector_new[a] = numerator / denominator
        
        return prob_vector_new

    def min_total_loss(self, loss_vectors: list, actions: list):
        '''
        provided with a list of loss vectors for all rounds, returns the action that would
        have given minimall loss together its loss value
        '''
        # TODO: improve efficiency (maintain total regrets per action and update each round)
        min = float('inf')
        action = -1
        for i in range(len(actions)):
            total_loss = 0
            for loss_vector in loss_vectors:
                total_loss += self.loss(loss_vector, actions[i])
            if total_loss < min:
                min = total_loss
                action = i
        return action, min

    def total_loss_vector(self, total_loss_vector: list, loss_vector: list, actions: list):
        '''
        provided with the loss vector for a round and the latest total loss vector, updates 
        the totall loss value for each action and returns this
        '''
        for i in range(len(actions)):
            total_loss_vector[i] += self.loss(loss_vector, actions[i])

        return total_loss_vector


    def loss(self, loss_vector: list, path: list):
        '''
        computes the instantaneous loss based on a loss_vector and a binary encoded path
        '''
        return np.inner(loss_vector, path)
    
    def cov_matrix(self, prob_vector: list, actions: list):
        '''
        computes and returns the covariance matrix based on a probability vector and a
        list of actions (list of binary encoded actions)
        '''
        cov_matrix = 0
        for a in range(len(actions)):
            square_matrix = np.outer(actions[a], actions[a])
            cov_matrix += prob_vector[a] * square_matrix
        return cov_matrix

    def estimate_loss_vector(self, prob_vector: list, actions: list, action: int, instantaneous_loss: int):
        '''
        computes and resturns an estimate loss vector based on a probability vector, a list of actions,
        the index of the chosen action, and the instantaneous loss
        '''
        pinv_cov_matrix = np.linalg.pinv(self.cov_matrix(prob_vector, actions))
        estimated_loss_vector = pinv_cov_matrix @ actions[action] * instantaneous_loss
        return estimated_loss_vector
    
    def actions_with_edge(self, actions: list):
        actions_edges_dict = {}
        for i in range(len(actions[-1])):
            actions_with_edge = []
            for j in range(len(actions)):
                if actions[j][i] == 1:
                    actions_with_edge.append(j)
            actions_edges_dict[i] = actions_with_edge
        return actions_edges_dict

    def loss_values_semi(self, action: list, loss_vector: list):
        semi_losses = [0] * len(action)
        for i in action:
            if action[i] == 1:
                semi_losses[i] = loss_vector[i]
        return semi_losses

    def estimate_loss_vector_semi(self, prob_vector: list, actions: list, action: int, semi_losses: int, actions_edges_dict: dict):
        '''
        computes and resturns an estimate loss vector based on a probability vector, a list of actions,
        the index of the chosen action, and the instantaneous loss
        '''
        estimated_loss_vector = [0] * self.graph.num_edges()
        action = actions[action]
        for i in range(len(action)):
            if action[i] == 1:
                actions_with_edge = actions_edges_dict[i]
                prob_sum = 0
                for action_with_edge in actions_with_edge:
                    prob_sum += prob_vector[action_with_edge]
                z_estimate = (semi_losses[i] / prob_sum)
                estimated_loss_vector[i] = z_estimate
        
        return estimated_loss_vector

    def exp2(self, eta: float, actions: list, rounds: int, game='full', debug=False):
        '''
        execute the exp2 algorithm and return the total regret, more information is printed into
        the terminal during execution
        '''
        print("running exp2 algo (setting: " + game + ")...")

        # initialize probability vector
        prob_vector = [1/len(actions)] * len(actions)
        losses_sum = 0
        progress = 0
        total_loss_vectors = [[0] * len(actions)]
        expected_losses = []
        expected_losses_best_action = []
        total_regrets = []
        actions_edges_dict = None

        if game == 'semi':
            actions_edges_dict = self.actions_with_edge(actions)

        # loop through the rounds
        for t in range(1, rounds+1):
            if t % (rounds/10) == 0:
                progress += 10
                print('round progress: ' + str(progress) + '%')

            self.graph.update_edge_weights()                            # "adversary" generates new edge weights

            # choose an action
            action = self.action(prob_vector)                         # decision maker chooses action based on probability vector P_t

            loss_vector_full = self.graph.get_all_edge_weights()

            instantaneous_loss = self.loss(loss_vector_full, actions[action])
            losses_sum += instantaneous_loss

            # if bandit game, reset the loss vector to an estimated version based on the instantaneous loss
            if game == 'bandit':
                loss_vector = self.estimate_loss_vector(prob_vector, actions, action, instantaneous_loss)
            elif game == 'semi':
                semi_losses = self.loss_values_semi(actions[action], loss_vector_full)
                loss_vector = self.estimate_loss_vector_semi(prob_vector, actions, action, semi_losses, actions_edges_dict)
            else:
                loss_vector = loss_vector_full

            if debug:
                print("\nround " + str(t) + " ↓")
                print("prob_vector =\t\t" + str(np.round(prob_vector, 2)))
                print("action =\t\t" + str(action))

            total_loss_vectors.append(self.total_loss_vector(total_loss_vectors[-1], loss_vector_full, actions))
            expected_losses.append(losses_sum / t)
            expected_loss_best_action = (min(total_loss_vectors[-1]) / t)
            #expected_losses_best_action.append(total_loss_best_action / t)
            total_regrets.append(expected_losses[-1] - expected_loss_best_action)

            # calculate P_t+1
            prob_vector = self.update_prob_vector(prob_vector, loss_vector, eta, actions)

        print("\ncalculating total regret...")

        expected_loss = expected_losses[-1]
        expected_loss_best_action = expected_loss_best_action
        
        print("\nfinal prob_vector =\t\t\t\t" + str(np.round(prob_vector, 2)))
        best_action = total_loss_vectors[-1].index(min(total_loss_vectors[-1]))

        print("best action (in hindsight) =\t\t\t" + str(best_action))

        print("\nexpected loss =\t\t\t\t\t" + str(expected_loss))
        print("expected loss for best action overall =\t\t" + str(expected_loss_best_action))

        total_regret = total_regrets[-1]

        print("\ntotal regret =\t\t\t\t\t" + str(total_regret))

        return total_regrets
