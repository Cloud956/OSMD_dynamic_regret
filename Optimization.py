from math import log
import numpy as np
from scipy.optimize import Bounds
import scipy.optimize as spo


def osmd_pre(a):
    A=a
    A=np.transpose(A)
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
        print(result.x)
        print(np.dot(A,result.x))

    x = np.dot(A,result.x)
    A_inv = np.linalg.pinv(A)
    pt = np.dot(A_inv,x)
    pt =[round(i,6) for i in pt]
    print(pt)
    print(sum(pt))
