import numpy as np
import matplotlib.pyplot as plt
from Optimize import Optimizer
from Optimize import SGD

def f(x):
    return x*np.cos(x)
x = np.linspace(0,15,100)
xl = 0
xr = 15
x0 = np.random.choice(x)
lrs = [0.01, 0.1, 0.5]
for lr in lrs:
    optim = SGD(lr = lr, root = "SGD_output")
    optim.run(f, xl,xr, x0)
