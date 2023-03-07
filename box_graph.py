import sys
from point import Point
from algorithms import Algos
import matplotlib.pyplot as plt
def do_setup():
    print("Input the desired size of the matrix (Height,Length)")
    height,length = eval(input())

    print("Enter start point coords -> a,b")
    sx, sy = eval(input())
    start = Point(sx, sy)
    if sx < 0 or sy < 0 or sx > height or sy > length:
        print("incorrect input")
        sys.exit(2)
    print("Enter goal point coords -> a,b")
    gx, gy = eval(input())
    goal = Point(gx, gy)
    if gx < 0 or gy < 0 or gx > height or gy > length:
        print("incorrect input")
        sys.exit(2)
    min_length = goal.sum()-start.sum()
    print(f"Enter desired length of path (Valid lengths are {min_length} + 2*n)")
    path_len = eval(input())
    print("Enter desired d bound for the cost vector")
    cost_bound = eval(input())
    print("Enter the desired learning rate")
    learning_rate = eval(input())

    algo = Algos(height,length, path_len,start,goal,cost_bound,learning_rate)
    algo.init_paths()
    algo.initialize_probabilities()
    return algo

def do_rounds(algo,turns):
    regrets = []
    for i in range(turns):
       # algo.regenerate_cost()
        choice,index = algo.make_a_choice()
        loss = algo.get_loss(choice)
        regret = algo.normal_regret(choice)
        regrets.append(regret)
        algo.exp2_main()
   # b=2
    plt.plot(regrets)
    plt.show()
    #print(loss)
if __name__ == '__main__':
    algo = do_setup()
    print("Enter T")
    T=eval(input())
    do_rounds(algo,T)