from point import Point, Path, Edge
from copy import deepcopy
import numpy as np


class Algos:
    def __init__(self, height,length, path_len,start,goal):
        self.path_len = path_len
        self.edges_max = (2 * height*length) - height-length
        self.h=height
        self.l=length
        self.start=start
        self.goal=goal
    def points_to_index(self, p1, p2):
        x1, y1 = p1.xy()
        x2, y2 = p2.xy()
        if x1 == x2:  # left-right
            ylow = min(y2, y1)
            return (self.l - 1) * (x1 - 1) + (ylow - 1)
        elif y1 == y2:
            xlow = min(x1, x2)
            return (self.h - 1) * (y1 - 1) + (xlow - 1) + (self.h * (self.l - 1))

    def is_valid_point(self, x,y):
        return 0 < x < (self.h + 1) and 0 < y < (self.l+1)

    def point_can_reach(self,point,depth):
        min_distance = self.goal.sum() - point.sum()
        remaining_moves = self.path_len - depth - 1
        rest = remaining_moves - min_distance
        if rest % 2 == 0 and rest >= 0:
            return True
        return False
    def find_point_neighbours(self, point,depth):
        neighbours = []
        x, y = point.xy()
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):
                if (a == x) != (b == y) and self.is_valid_point(a, b):
                    neighbours.append(Point(a, b))
        return [i for i in neighbours if self.point_can_reach(i, depth)]

    def get_next_paths(self, path):
        children = []
        past = path.path
        last_point = path.end
        neighbours = self.find_point_neighbours(last_point,len(path))
        for neighbour in neighbours:
            new_edge = Edge(self.points_to_index(last_point, neighbour), last_point, neighbour)
            if new_edge not in past:
                past_copy = deepcopy(past)
                past_copy.append(new_edge)
                children.append(Path(past_copy, neighbour))
        return children

    def bfs_path(self):
        current_wave = [Path([],self.start)]
        for i in range(self.path_len):
            next_wave = []
            for p in current_wave:
                children = self.get_next_paths(p)
                next_wave.extend(children)
            current_wave = next_wave
        return [i for i in current_wave if i.end == self.goal]

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
