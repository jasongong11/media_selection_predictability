from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar
import numpy as np

def func(Pmax, S, n):
    return abs(S +Pmax * np.log2(Pmax) + (1-Pmax) * np.log2(1-Pmax) - (1-Pmax) * np.log2(n-1))

def predictability(S,n):
    root = minimize_scalar(func, bounds = (0,1), method='bounded', args=(S,n))
    return root.x