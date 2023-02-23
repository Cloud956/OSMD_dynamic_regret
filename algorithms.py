from point import Point
from point import Path
from copy import deepcopy
import numpy as np


class Algos:
    def __init__(self, size, path_len):
        self.path_len = path_len
        self.n = size
        self.edges_max = 2 * pow(size, 2) - 2 * size

    def points_to_index(self, p1, p2):
        x1, y1 = p1.xy()
        x2, y2 = p2.xy()
        if x1 == x2:  # left-right
            ylow = min(y2, y1)
            return (self.n - 1) * (x1 - 1) + (ylow - 1)
        elif y1 == y2:
            xlow = min(x1, x2)
            return (self.n - 1) * (y1 - 1) + (xlow - 1) + (self.n * (self.n - 1))

    def is_valid_point(self, index):
        return 0 < index < (self.n + 1)

    def find_point_neighbours(self, point):
        neighbours = []
        x, y = point.xy()
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):
                if (a == x) != (b == y) and self.is_valid_point(a) and self.is_valid_point(b):
                    neighbours.append(Point(a, b))
        return neighbours

    def get_next_paths(self, path):
        children = []
        past = path.path
        last_point = past[-1]
        neighbours = self.find_point_neighbours(last_point)
        for neighbour in neighbours:
            if neighbour not in past:
                past_copy = deepcopy(past)
                past_copy.append(neighbour)
                children.append(Path(past_copy))
        return children

    def bfs_path(self, start, goal):
        current_wave = [Path([start])]
        for i in range(self.path_len):
            next_wave = []
            for p in current_wave:
                children = self.get_next_paths(p)
                next_wave.extend(children)
            current_wave = next_wave
        return [i for i in current_wave if i.path[-1] == goal]

    def paths_to_boolean(self, paths):
        b_paths = []
        for p in paths:
            path = p.path
            length = len(p)
            curr = np.zeros(self.edges_max)
            for i in range(length):
                index = self.points_to_index(path[i], path[i + 1])
                curr[index] = 1
            b_paths.append(curr)
        return b_paths
