from math import log
import numpy as np
from scipy.optimize import Bounds
import scipy.optimize as spo
from math import exp

def osmd_pre(a):
    A = a
    A = np.transpose(A)
    def f(w):
        x = np.dot(A,w)
        y=[]
        for xi in x:
            y.append(xi*log(xi))
        return sum(y)

    const = ({'type': 'eq','fun': lambda w : sum(w) - 1})
    bounds = Bounds(lb=0,keep_feasible=True)
    initial_guess = [1/len(a)] * len(a)
    w_start = np.array(initial_guess)
    result = spo.minimize(f,w_start,options={'disp':True},constraints=const,bounds=bounds)
    if result.success:
        print("worked")
        print("w vector ", result.x)
        print("x vector ",np.dot(A,result.x))

    x = np.dot(A,result.x)
    A_inv = np.linalg.pinv(A)
    pt = np.dot(A_inv,x)
    pt =[round(i,6) for i in pt]
    print(pt)
    print(sum(pt))
def osmd_middle(x,learning,loss):
    new_w = x*exp(-1*learning*loss)
    return new_w
def osmd_big_guy(new_w,A):
    pass
    a=A
    A = a
    A = np.transpose(A)
    wt = new_w
    def f(input_w):
        x = np.dot(A,input_w)
        res = 0
        for i in range(len(x)):
            xi = x[i]
            yi = wt[i]
            res+=xi*log(xi/yi)
            res+=yi-xi
        return res

    const = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
    bounds = Bounds(lb=0, keep_feasible=True)
    initial_guess = [1 / len(a)] * len(a)
    w_start = np.array(initial_guess)
    result = spo.minimize(f, w_start, options={'disp': True}, constraints=const, bounds=bounds)
    if result.success:
        print("worked")
        print("w vector ", result.x)
        print("x vector ", np.dot(A, result.x))


a=[[0,1,0,1],[1,0,1,0],[1,1,0,0]]
osmd_pre(a)