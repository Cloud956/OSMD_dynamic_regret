import sys
from Point import Point

print("Input the desired size of the matrix")

size = eval(input())

edges_max = 2*pow(size,2) - 2*size
print(edges_max)
print("Enter start point coords -> a,b")
sx,sy = eval(input())
start = Point(sx,sy)
if sx<0 or sy<0 or sx>size or sy>size:
    print("incorrect input")
    sys.exit(2)
print("Enter goal point coords -> a,b")
gx,gy = eval(input())
goal = Point(gx,gy)
if gx<0 or gy<0 or gx>size or gy>size:
    print("incorrect input")
    sys.exit(2)


print(start)
print(goal)