from point import Point, Path, Edge
from copy import deepcopy
import numpy as np
import random
from math import exp

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
    def points_to_index(self, p1, p2):
        x1, y1 = p1.xy()
        x2, y2 = p2.xy()
        if x1 == x2:  # left-right
            ylow = min(y2, y1)
            return (self.l - 1) * (x1 - 1) + (ylow - 1)
        elif y1 == y2:
            xlow = min(x1, x2)
            return (self.h - 1) * (y1 - 1) + (xlow - 1) + (self.h * (self.l - 1))

    def is_valid_point(self, x, y):
        return 0 < x < (self.h + 1) and 0 < y < (self.l + 1)

    def point_can_reach(self, point, depth):
        min_distance = self.goal.sum() - point.sum()
        remaining_moves = self.path_len - depth - 1
        rest = remaining_moves - min_distance
        if rest % 2 == 0 and rest >= 0:
            return True
        return False

    def initialize_probabilities(self):
        number = len(self.paths)
        self.probabilities = [1/number]*number

    def make_a_choice(self):
        probs = self.probabilities
        roll = random.random()
        total=0
        for i in range(len(probs)):
            total += probs[i]
            if roll <= total:
                #print(f"ROLLED {roll} RETURNING {i}")
                return self.bpath[i],i
    def get_loss(self,choice):
        #print(choice)
       # print(self.cost)
        return np.dot(self.cost,np.transpose(choice))
    def min_loss(self):
        nums=[]
        for p in self.bpath:
            nums.append(np.dot(self.cost,p))
        return min(nums)
    def normal_regret(self,choice):
        minimal = self.min_loss()
        return np.dot(self.cost,choice)-minimal
    def find_point_neighbours(self, point, depth):
        neighbours = []
        x, y = point.xy()
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):
                if (a == x) != (b == y) and self.is_valid_point(a, b):
                    neighbours.append(Point(a, b))
        return [i for i in neighbours if self.point_can_reach(i, depth)]
    def init_paths(self):
        path = self.bfs_path()
        self.paths = path
        self.bpath = self.paths_to_boolean(path)
    def get_next_paths(self, path):
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
        nums = []
        for i in range(self.edges_max):
            nums.append(random.random())
        multiplication = int(self.cost_bound / 0.5)
        nums = [multiplication * (i - 0.5) for i in nums]
        self.cost = nums

    def bfs_path(self):
        current_wave = [Path([], self.start)]
        for i in range(self.path_len):
            next_wave = []
            for p in current_wave:
                children = self.get_next_paths(p)
                next_wave.extend(children)
            current_wave = next_wave
        return [i for i in current_wave if i.end == self.goal]
    def exp2_main(self):
        probs = []
        bot = self.exp2_bot()
        for i in range(len(self.bpath)):
            path=self.bpath[i]
            top= self.exp2_top(path)
            probs.append(self.probabilities[i]*top/bot)
        #print(probs)
        #print(sum(probs))
        self.probabilities = probs

    def run_semi_bandit(self):
        b=2

    def run_bandit(self,loss,choice):
        P=0
        for i in range(len(self.bpath)):
            path=self.bpath[i]
            normal_path = np.array(path)
            mult = np.outer(normal_path,normal_path)
            P+= mult*self.probabilities[i]
        P_plus = np.linalg.pinv(P)
        cost = np.matmul(P_plus,choice) * loss
        self.cost = cost
    def exp2_top(self,choice):
        learn_var = -1*self.learn
        top = exp(learn_var * np.dot(self.cost,choice))
        return top
    def exp2_bot(self):
        learn_var = -1 * self.learn
        bot = 0
        zT = np.array(self.cost).transpose()
        for i in range(len(self.bpath)):
            bT = np.array(self.bpath[i]).transpose()
            addition = exp(learn_var * np.dot(bT, zT)) * self.probabilities[i]
            bot += addition
        return bot
    def paths_to_boolean(self, paths):
        b_paths = []
        for p in paths:
            path = p.path
            curr = np.zeros(self.edges_max)
            for edge in path:
                index = edge.index
                curr[index] = 1
            b_paths.append(curr)
        return b_paths
