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

    def __str__(self):
        return f"Point at x: {self.x} y:{self.y}"

    def __repr__(self):
        return str(self)


class Path:
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return len(self.path) - 1
