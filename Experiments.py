import matplotlib.pyplot as plt
from tqdm import tqdm
from Settings import *
from algorithms import *
import os
from os.path import join
import pandas as pd
from game_runner import do_rounds,do_rounds_osmd, do_rounds_random
import time
information_dict={
    1: "full",
    2: "semi",
    3: "bandit"
}
def plot(total_results_exp,total_results_osmd,total_results_random,name,num_of_runs,low_bound,high_bound):
    exp2, = plt.plot(total_results_exp, label='EXP2')
    osmd, = plt.plot(total_results_osmd, label='OSMD')
    random, = plt.plot(total_results_random, label='Random')
    plt.title(f'Dynamic regret - {name}, {num_of_runs} runs.')
    plt.ylim([low_bound, high_bound])
    plt.legend(handles=[exp2, osmd, random])
    plt.xlabel('Round number')
    plt.ylabel('Dynamic regret value')
    plt.show()

def save_results(data,name):
    os.makedirs('data',exist_ok=True)
    csv_path = join('data',name+'.csv')
    df = pd.DataFrame(data,columns=['regret_value'])
    df.to_csv(csv_path,index=False)
def save_time_results(time,name):
    txt_path = join('data',name+'.txt')
    f = open(txt_path,'w+')
    f.write(str(time))
    f.close()

def run_main_experiments(num_of_runs,rounds, setting,information_setting,cost_mode,experiment_name):
    print("Pick a gamemode!  \n"
          "1 - Full Information \n"
          "2 - Semi-Bandit \n"
          "3 - Full Bandit")
    print("Select the algorithm! \n"
          "1 --> EXP2 \n"
          "2 --> OSMD \n"
          "3 --> RANDOM\n")
    information_name = information_dict.get(information_setting)
    setting.set_cost_mode(cost_mode)
    total_results_exp, time_exp = run_single_experiment(num_of_runs,setting,1,
                                          information_setting,rounds)
    total_results_osmd, time_osmd = run_single_experiment(num_of_runs,setting,2,
                                         information_setting,rounds)
    total_results_random, time_random = run_single_experiment(num_of_runs,setting,3,
                                         information_setting,rounds)
    save_results(total_results_exp,'exp_'+experiment_name+"_"+information_name)
    save_results(total_results_osmd, 'osmd_' + experiment_name+"_"+information_name)
    save_results(total_results_random, 'random_' + experiment_name+ "_"+information_name)
    save_time_results(time_exp,'exp_'+experiment_name+"_"+information_name)
    save_time_results(time_osmd, 'osmd_' + experiment_name + "_" + information_name)
    save_time_results(time_random, 'random_' + experiment_name + "_" + information_name)
def run_single_experiment(num_of_runs,setting,algorithm, information_setting,rounds):
    total_results = []
    failed=0
    total_times=0
    for i in tqdm(range(num_of_runs)):
        try:
            start = time.time()
            result_from_here = run_experiment(algorithm,information_setting,rounds,setting)
            end = time.time()
            total_times+=(end-start)
            if total_results:
                total_results = [x+y for x,y in zip(total_results,result_from_here)]
            else:
                total_results = result_from_here
        except:
            failed+=1
    total_results = [x / (num_of_runs-failed) for x in total_results]
    total_times = total_times/(num_of_runs-failed)
    total_times = round(total_times,3)
    return total_results, total_times
def run_experiment(algorithm, information_setting,rounds,setting):
    algo = Algos(setting)
    algo.init_paths()
    algo.initialize_probabilities()
    if algorithm == 1:
        regrets = do_rounds(algo,rounds,information_setting)
    elif algorithm == 2:
        regrets = do_rounds_osmd(algo,rounds,information_setting)
    else:
        regrets = do_rounds_random(algo,rounds,information_setting)
    return regrets

def run_first_experiment():
    setting = Setting()
    setting.set_setting('experiment_1')
    setting.set_learn_exp(0.04)
    setting.set_learn_osmd(0.086)
    for i in range(1,4):
        run_main_experiments(100,400,setting,i,'uniform','experiment_1')
def run_third_experiment():
    setting = Setting()
    setting.set_setting('experiment_1')
    setting.set_learn_exp(0.021)
    setting.set_learn_osmd(0.046)
    for i in range(1, 4):
        run_main_experiments(100, 1400, setting, i, 'uniform_NOT', 'experiment_3')
def run_second_experiment():
    setting = Setting()
    setting.set_setting('experiment_2')
    setting.set_learn_exp(0.037)
    setting.set_learn_osmd(0.073)
    for i in range(1,4):
        run_main_experiments(100,400,setting,i,'uniform_NOT','experiment_2')
def plot_experiments(experiment,graph,loss,low,up,num_of_runs):
    folder = 'data'
    exp2 = pd.read_csv(join(folder, f'exp_{experiment}_full.csv'))
    osmd = pd.read_csv(join(folder, f'osmd_{experiment}_full.csv'))
    random = pd.read_csv(join(folder, f'random_{experiment}_full.csv'))
    plot(exp2, osmd, random, f'{graph}, {loss}, full information', num_of_runs, low, up)
    exp2 = pd.read_csv(join(folder, f'exp_{experiment}_semi.csv'))
    osmd = pd.read_csv(join(folder, f'osmd_{experiment}_semi.csv'))
    random = pd.read_csv(join(folder, f'random_{experiment}_semi.csv'))
    plot(exp2, osmd, random, f'{graph}, {loss}, semi-bandit', num_of_runs, low, up)
    exp2 = pd.read_csv(join(folder, f'exp_{experiment}_bandit.csv'))
    osmd = pd.read_csv(join(folder, f'osmd_{experiment}_bandit.csv'))
    random = pd.read_csv(join(folder, f'random_{experiment}_bandit.csv'))
    plot(exp2, osmd, random, f'{graph}, {loss}, bandit', num_of_runs, low, up)
if __name__ == '__main__':
    #plot_experiments('experiment_1','small graph','uniform loss',1.1,1.3,100)
    plot_experiments('experiment_2', 'big graph', 'shifted loss',1.6,2,100)
    #plot_experiments('experiment_3', 'small graph', 'shifted loss', 0.65, 1.1, 100)
    #run_first_experiment()
   # run_third_experiment()
    #run_second_experiment()




