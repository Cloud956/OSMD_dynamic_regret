class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, Point):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def xy(self):
        return [self.x, self.y]

    def sum(self):
        return self.x + self.y

    def __str__(self):
        return f"Point at x: {self.x} y:{self.y}"

    def __repr__(self):
        return str(self)


class Path:
    def __init__(self, path,last_point):
        self.path = path
        self.end = last_point
    def __len__(self):
        return len(self.path)
class Edge:
    def __init__(self,index,point_from,point_into):
        self.index = index
        self.source = point_from
        self.target = point_into
    def __eq__(self,other):
        if not isinstance(other,Edge):
            return NotImplemented
        return self.index == other.index

    def __str__(self):
        return f"Edge from {self.source} to {self.target}"

    def __repr__(self):
        return str(self)