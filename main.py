import sys
from point import Point
from algorithms import Algos
import matplotlib.pyplot as plt
from tqdm import tqdm
def do_setup():
    print("Enter s to skip a bit of a setup or press ENTER to move forward")
    cheat = input()
    if cheat == "s":
        height,length,sx,sy,gx,gy,path_len,learning_rate,cost_bound = 4,4,1,1,3,4,7,0.1,2
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
        regret = algo.dynamic_regret(choice)
        regrets.append(regret)
        if mode == 2:
            algo.run_semi_bandit(choice)
        elif mode ==3:
            algo.run_bandit(loss,choice)


        algo.exp2_main()

    over_time_val=[]
    sum_val=0
    for i in range(len(regrets)):
        sum_val+=regrets[i]
        over_time_val.append(sum_val/i)
    return over_time_val
def do_rounds_osmd(algo,turns,mode):
    regrets = []
    algo.osmd_pre()
    if mode == 2:
        algo.precompute_semi_bandit()
    for i in range(turns):
        algo.regenerate_cost()
        algo.set_osmd_pt()
        choice,index = algo.make_a_choice()

        regret = algo.dynamic_regret(choice)
        regrets.append(regret)

        if mode == 2:
            algo.run_semi_bandit(choice)
        elif mode ==3:
            loss = algo.get_loss(choice)
            algo.run_bandit(loss,choice)

        algo.run_osmd()

    over_time_val=[]
    sum_val=0
    for i in range(len(regrets)):
        sum_val+=regrets[i]
        over_time_val.append(sum_val/i)
    return over_time_val
def do_rounds_random(algo,turns,mode):
    regrets = []
    for i in range(turns):
        algo.regenerate_cost()
        choice,index = algo.make_a_choice()
        regret = algo.dynamic_regret(choice)
        regrets.append(regret)
        if mode == 2:
            algo.run_semi_bandit(choice)
        elif mode ==3:
            loss = algo.get_loss(choice)
            algo.run_bandit(loss,choice)


    over_time_val=[]
    sum_val=0
    for i in range(len(regrets)):
        sum_val+=regrets[i]
        over_time_val.append(sum_val/i)
    return over_time_val
def run_main_experiments(num_of_runs,start, goal, cost, path_length, epsilon, information_setting,rounds,map_height,map_length,name):
    print("Pick a gamemode!  \n"
          "1 - Full Information \n"
          "2 - Semi-Bandit \n"
          "3 - Full Bandit")
    print("Select the algorithm! \n"
          "1 --> EXP2 \n"
          "2 --> OSMD \n"
          "3 --> RANDOM\n")
    total_results_exp = run_single_experiment(num_of_runs,start, goal, cost, path_length, epsilon,1,
                                          information_setting,rounds,map_height,map_length)
    total_results_osmd = run_single_experiment(num_of_runs,start, goal, cost, path_length, epsilon,2,
                                          information_setting,rounds,map_height,map_length)
    total_results_random = run_single_experiment(num_of_runs,start, goal, cost, path_length, epsilon,3,
                                          information_setting,rounds,map_height,map_length)
    exp2, = plt.plot(total_results_exp, label='EXP2')
    osmd, = plt.plot(total_results_osmd, label='OSMD')
    random, = plt.plot(total_results_random, label='Random')
    plt.title(f'Dynamic regret - {name}, {num_of_runs} runs.')
    plt.ylim([0,4])
    plt.legend(handles=[exp2,osmd,random])
    plt.xlabel('Round number')
    plt.ylabel('Dynamic regret value')
    plt.show()


def run_single_experiment(num_of_runs,start, goal, cost, path_length, epsilon,algorithm, information_setting,rounds,map_height,map_length):
    total_results = []
    failed=0
    for i in tqdm(range(num_of_runs)):
        try:
            result_from_here = run_experiment(start, goal, cost, path_length, epsilon,algorithm,
                                          information_setting,rounds,map_height,map_length)
            if total_results:
                total_results = [x+y for x,y in zip(total_results,result_from_here)]
            else:
                total_results = result_from_here
        except:
            failed+=1
    total_results = [x / num_of_runs-failed for x in total_results]
    return total_results
def run_experiment(start, goal, cost, path_length, epsilon,algorithm, information_setting,rounds,map_height,map_length):

    algo = Algos(map_height,map_length,path_length,start,goal,cost,epsilon)
    algo.set_cost_mode(0)
    algo.init_paths()
    algo.initialize_probabilities()
    if algorithm == 1:
        regrets = do_rounds(algo,rounds,information_setting)
    elif algorithm == 2:
        regrets = do_rounds_osmd(algo,rounds,information_setting)
    else:
        regrets = do_rounds(algo,rounds,information_setting)
    return regrets
if __name__ == '__main__':
    height, length, sx, sy, gx, gy, path_len, learning_rate, cost_bound = 4, 4, 1, 1, 3, 4, 7, 0.1, 2
    start_point = Point(sx,sy)
    goal_point = Point(gx,gy)
    run_main_experiments(40,start_point,goal_point,cost_bound,path_len,learning_rate,3,500,height,length,"Bandit")
    run_main_experiments(1000, start_point, goal_point, cost_bound, path_len, learning_rate, 2, 500, height, length,
                         "Semi-Bandit")
    run_main_experiments(1000, start_point, goal_point, cost_bound, path_len, learning_rate, 1, 500, height, length,
                         "Full Information")


