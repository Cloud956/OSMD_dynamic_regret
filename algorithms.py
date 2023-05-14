from point import Point, Path, Edge
from copy import deepcopy
import numpy as np
import random
from math import exp
from math import log
import numpy as np
from scipy.optimize import Bounds
import scipy.optimize as spo


class Algos:
    def __init__(self, height, length, path_len, start, goal, cost_bound,learning_rate):
        self.path_len = path_len
        self.edges_max = (2 * height * length) - height - length
        self.h = height
        self.l = length
        self.start = start
        self.goal = goal
        self.cost_bound = cost_bound
        self.cost = None
        self.probabilities = None
        self.paths = None
        self.learn = learning_rate
        self.bpath=None
        self.regenerate_cost()
        self.semi_bandit_dict=None
    def points_to_index(self, p1, p2):
        """
        :param p1: Point from which the edge is coming out of
        :param p2: Point into which the edge is coming    (p1 and p2 are connected by edge)
        :return: The index number ( in the boolean array) of the edge from point p1 to point p2
        """
        x1, y1 = p1.xy()
        x2, y2 = p2.xy()
        if x1 == x2:  # left-right
            ylow = min(y2, y1)
            return (self.l - 1) * (x1 - 1) + (ylow - 1)
        elif y1 == y2:
            xlow = min(x1, x2)
            return (self.h - 1) * (y1 - 1) + (xlow - 1) + (self.h * (self.l - 1))

    def is_valid_point(self, x, y):
        """
        :param x: x value of point
        :param y: y value of point
        :return: Whether the point is in the grid graph or not
        """
        return 0 < x < (self.h + 1) and 0 < y < (self.l + 1)

    def point_can_reach(self, point, depth):
        """
        :param point: Point which we are currently examining
        :param depth: The remaining amount of edges we can possibly use to reach the goal
        :return:  Whether from the Point point it is possible to reach the goal point
            Why? -> If it cannot reach goal, we can ignore this path and save computation for the bfs algorithm
        """
        min_distance = self.goal.sum() - point.sum()
        remaining_moves = self.path_len - depth - 1
        rest = remaining_moves - min_distance
        if rest % 2 == 0 and rest >= 0:
            return True
        return False

    def initialize_probabilities(self):
        """
        Initializes the probs for actions in A. All p(a) = 1/|A|
        """
        number = len(self.paths)
        self.probabilities = [1/number]*number

    def make_a_choice(self):
        """
        Decides on a choice of an action a using all of the probabilities.
        :return: Action a in boolean vertex form and the index of that action a in the list of actions A
        """
        probs = self.probabilities
        roll = random.random()
        total=0
        for i in range(len(probs)):
            total += probs[i]
            if roll <= total:
                #print(f"ROLLED {roll} RETURNING {i}")
                return self.bpath[i],i
    def get_loss(self,choice):
        """
        :param choice:  boolean vertex of chosen action
        :return: loss incurred by selecting this action
        """
        #print(choice)
       # print(self.cost)
        return np.dot(self.cost,np.transpose(choice))
    def min_loss(self):
        """
        :return: Minimal loss you could have incurred.
        """
        nums=[]
        for p in self.bpath:
            nums.append(np.dot(self.cost,p))
        return min(nums)
    def dynamic_regret(self, choice):
        """
        :param choice: The action you have selected (boolean vector)
        :return: Your dynamic regret at a given timestep ( function to be used before refreshing the loss vector)
        """
        minimal = self.min_loss()
        return np.dot(self.cost,choice)-minimal
    def find_point_neighbours(self, point, depth):
        """
        :param point: Point currently being inspected
        :param depth: Remaining amount of moves you can do
        :return: All the neighbours of Point point, who lead to paths which do not reuse edges and can possibly
                                                                                                reach the goal point
        """
        neighbours = []
        x, y = point.xy()
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):
                if (a == x) != (b == y) and self.is_valid_point(a, b):
                    neighbours.append(Point(a, b))
        return [i for i in neighbours if self.point_can_reach(i, depth)]
    def init_paths(self):
        """
        Initializes the paths, runs the bfs algorithm.
        """
        path = self.bfs_path()
        self.paths = path
        self.bpath = self.paths_to_boolean(path)
    def get_next_paths(self, path):
        """
        :param path: Path being inspected
        :return: All the paths which can be reached by extending the starting path, which can reach the goal.
        """
        children = []
        past = path.path
        last_point = path.end
        neighbours = self.find_point_neighbours(last_point, len(path))
        for neighbour in neighbours:
            new_edge = Edge(self.points_to_index(last_point, neighbour), last_point, neighbour)
            if new_edge not in past:
                past_copy = deepcopy(past)
                past_copy.append(new_edge)
                children.append(Path(past_copy, neighbour))
        return children

    def regenerate_cost(self):
        """
        Draws a new loss vector
        """
        nums = []
        for i in range(self.edges_max):
            nums.append(random.random())
        multiplication = int(self.cost_bound / 0.5)
        nums = [multiplication * (i - 0.5) for i in nums]
        self.cost = nums

    def bfs_path(self):
        """
        Runs the bfs algorithm to find all paths from start to goal
        """
        current_wave = [Path([], self.start)]
        for i in range(self.path_len):
            next_wave = []
            for p in current_wave:
                children = self.get_next_paths(p)
                next_wave.extend(children)
            current_wave = next_wave
        return [i for i in current_wave if i.end == self.goal]
    def exp2_main(self):
        """
        Updates probabilities according to the exp2 algorithm
        """
        probs = []
        bot = self.exp2_bot()
        for i in range(len(self.bpath)):
            path=self.bpath[i]
            top= self.exp2_top(path)
            probs.append(self.probabilities[i]*top/bot)
        #print(probs)
        #print(sum(probs))
        self.probabilities = probs

    def run_semi_bandit(self,choice):
        coordinates = [a*b for a,b in zip(self.cost,choice)]
        end_cost = []
        for i in range(self.edges_max):
            val = coordinates[i]
            if val == 0:
                end_cost.append(0)
            else:
                containing = self.semi_bandit_dict.get(i)
                summer=0
                for a in containing:
                    summer+=self.probabilities[a]
                end_cost.append((val/summer)*choice[i])
        self.cost = end_cost


    def semi_bandit_check(self):
        for k,v in self.semi_bandit_dict.items():
            print(f"{k} : {len(v)}")
        for p in self.bpath:
            print(p)
    def precompute_semi_bandit(self):
        mapper = dict()
        for i in range(self.edges_max):
            mapper[i] = list()
        for j in range(len(self.bpath)):
            p = self.bpath[j]
            for i in range(self.edges_max):
                if p[i] == 1:
                    mapper[i].append(j)
        self.semi_bandit_dict = mapper

    def run_bandit(self,loss,choice):
        """
        :param loss: loss incurred
        :param choice: choice you made that turn
        :return: Nothing, updates the loss vector by the estimation according to a bandit game setting
        """
        P=0
        for i in range(len(self.bpath)):
            path=self.bpath[i]
            mult = np.outer(path,path)
            P+= mult*self.probabilities[i]
        P_plus = np.linalg.pinv(P)
        cost = np.matmul(P_plus,choice) * loss
        self.cost = cost
    def exp2_top(self,choice):
        """
        :param choice: Choice you made
        :return: The numerator for the exp2 algorithm
        """
        learn_var = -1*self.learn
        top = exp(learn_var * np.dot(self.cost,choice))
        return top
    def exp2_bot(self):
        """
        :return: The denominator for the exp2 algorithm, can be used once as it's shared for all the exp2 calculations
        """
        learn_var = -1 * self.learn
        bot = 0
        zT = np.array(self.cost).transpose()
        for i in range(len(self.bpath)):
            bT = np.array(self.bpath[i]).transpose()
            addition = exp(learn_var * np.dot(bT, zT)) * self.probabilities[i]
            bot += addition
        return bot


    def paths_to_boolean(self, paths):
        """
        :param paths: List of Path objects
        :return:  List of boolean vectors, which encode the Path objects
        """
        b_paths = []
        for p in paths:
            path = p.path
            curr = np.zeros(self.edges_max)
            for edge in path:
                index = edge.index
                curr[index] = 1
            b_paths.append(curr)
        return b_paths

    def osmd_pre(self):
        a = self.bpath
        A = a
        A = np.transpose(A)

        def f(w):
            x = np.dot(A, w)
            y = []
            for xi in x:
                y.append(xi * log(xi))
            return sum(y)

        const = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
        bounds = Bounds(lb=0, keep_feasible=True)
        initial_guess = [1 / len(a)] * len(a)
        w_start = np.array(initial_guess)
        result = spo.minimize(f, w_start, options={'disp': True}, constraints=const, bounds=bounds)
        if result.success:
            print("worked")
            print(result.x)
            print(np.dot(A, result.x))

        x = np.dot(A, result.x)
        A_inv = np.linalg.pinv(A)
        pt = np.dot(A_inv, x)
        #pt = [round(i, 2) for i in pt]
        print(pt)
        print(sum(pt))
        self.probabilities = pt