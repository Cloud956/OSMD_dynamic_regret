from algorithms import *
from Settings import *
setting = Setting()
setting.set_setting('experiment_1')
alg = Algos(setting)
alg.describe()
from math import e
import math
from numpy import log
def eta_exp(package):
    m,n,d = package
    top = 2 * (log((e*d)/m))
    bottom = n*m

    return math.sqrt((top/bottom))
def eta_osmd(package):
    m, n, d = package
    mid = m * log(d*m)
    divide = n*d
    big_part = mid/divide
    big = 2 *big_part
    return math.sqrt(big)
game_1 = (7,400,24)
game_2 = (10,400,60)
game_3 = (7,1400,24)
exp_1 = eta_exp(game_1)
exp_2 = eta_exp(game_2)
osmd_1 = eta_osmd(game_1)
osmd_2 = eta_osmd(game_2)
exp_3 = eta_exp(game_3)
osmd_3 = eta_osmd(game_3)
print(exp_1)
print(exp_2)
print(osmd_1)
print(osmd_2)

print('============')
print(exp_3)
print(osmd_3)