import sys
from point import Point
from algorithms import Algos
import matplotlib.pyplot as plt
def do_setup():
    print("Enter s to skip a bit of a setup or press ENTER to move forward")
    cheat = input()
    if cheat == "s":
        height,length,sx,sy,gx,gy,path_len,learning_rate,cost_bound = 3,3,1,1,3,3,6,0.1,3
    else:
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

        if gx < 0 or gy < 0 or gx > height or gy > length:
            print("incorrect input")
            sys.exit(2)
        goal = Point(gx, gy)
        min_length = goal.sum()-start.sum()
        print(f"Enter desired length of path (Valid lengths are {min_length} + 2*n)")
        path_len = eval(input())
        print("Enter desired d bound for the cost vector")
        cost_bound = eval(input())
        print("Enter the desired learning rate")
        learning_rate = eval(input())
    start = Point(sx, sy)
    goal = Point(gx, gy)
    print("Pick a gamemode!  \n"
          "1 - Full Information \n"
          "2 - Semi-Bandit \n"
          "3 - Full Bandit")
    gamemode=eval(input())
    algo = Algos(height,length, path_len,start,goal,cost_bound,learning_rate)
    algo.init_paths()
    algo.initialize_probabilities()
    return algo, gamemode

def do_rounds(algo,turns,mode):
    regrets = []
    if mode == 2:
        algo.precompute_semi_bandit()
    for i in range(turns):
        algo.regenerate_cost()
        choice,index = algo.make_a_choice()
        loss = algo.get_loss(choice)

        if mode == 2:
            algo.run_semi_bandit(choice)
        elif mode ==3:
            algo.run_bandit(loss,choice)

        regret = algo.dynamic_regret(choice)
        if regret > 20:
            b = 2
        regrets.append(regret)
        algo.exp2_main()

    over_time_val=[]
    sum_val=0
    for i in range(len(regrets)):
        sum_val+=regrets[i]
        over_time_val.append(sum_val/i)
    plt.plot(over_time_val)
    plt.title("Regrets / T over time")
    plt.show()
    plt.plot(regrets)
    plt.title('Regret over time')
    plt.show()

if __name__ == '__main__':
    algo,gamemode = do_setup()
    print("Enter T")
    T = eval(input())
    do_rounds(algo, T, gamemode)
    print("Final probabilities are:")
    print(algo.probabilities)