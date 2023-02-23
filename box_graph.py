import sys
from point import Point
from algorithms import Algos

print("Input the desired size of the matrix")

size = eval(input())

edges_max = 2 * pow(size, 2) - 2 * size
# print(edges_max)
print("Enter start point coords -> a,b")
sx, sy = eval(input())
start = Point(sx, sy)
if sx < 0 or sy < 0 or sx > size or sy > size:
    print("incorrect input")
    sys.exit(2)
print("Enter goal point coords -> a,b")
gx, gy = eval(input())
goal = Point(gx, gy)
if gx < 0 or gy < 0 or gx > size or gy > size:
    print("incorrect input")
    sys.exit(2)
print("Enter desired length of path")
path_len = eval(input())
algo = Algos(size, path_len)
print(start)
print(goal)
paths = algo.bfs_path(start, goal)
for p in paths:
    print(p.path)
b_paths = algo.paths_to_boolean(paths)
for p in b_paths:
    print(p)
