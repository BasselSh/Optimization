import numpy as np
import matplotlib.pyplot as plt
from Optimize import Optimizer


def f(x):
    return x*np.cos(x)
x = np.linspace(0,15,100)
x0 = np.random.choice(x)
lrs = [0.01, 0.1, 0.5]
for lr in lrs:
    optim = Optimizer(lr = lr)
    optim.run(f, x, x0)