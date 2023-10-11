import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
class Line:
    def __init__(self):
        self.m = None
        self.b = None

    def from_point_and_slope(self, x0, fx0, m):
        self.b = fx0-m*x0
        self.m = m
    def __call__(self,x):
        return self.m*x+self.b
    def get_x(self,y):
        return (y-self.b)/self.m

class Plotter:
    def __init__(self, h_fig=10, w_fig=10):
        self.fig = plt.figure(figsize=(w_fig, h_fig))
    
    def set_data(self, x, f):
        self.x = x
        self.f = f
        self.XMIN = np.min(x)
        self.XMAX = np.max(x)
        fx = f(x)
        rt = 0.2
        self.YLMIN = np.min(fx) - rt*np.abs(np.min(fx))
        self.YLMAX = np.max(fx) + rt*np.abs(np.max(fx))
    
    def plot(self, x, fx, color='blue'):
        fx_clipped = self.clipY(fx)
        plt.plot(x,fx_clipped, color = color)
    
    def scatter(self,x, fx, color):
        plt.scatter(x,fx, c=color)
    
    def clipY(self,y):
        if isinstance(y, np.ndarray):
            return np.where(y<=self.YLMIN, self.YLMIN,np.where(y>=self.YLMAX, self.YLMAX, y))
        if y>self.YLMAX:
            y = self.YLMAX
        if y<self.YLMIN:
            y = self.YLMIN
        return y
    
    def draw_line(self, x0,fx0,dfx0):
        line = Line()
        line.from_point_and_slope(x0,fx0,dfx0)
        def _correct_y_x(y_c):
            y = max(min(y_c, self.YLMAX), self.YLMIN)
            x = line.get_x(y)
            return x,y
        y_xmin = line(self.XMIN)
        x1,y1 = _correct_y_x(y_xmin)
        y_xmax = line(self.XMAX)
        x2,y2 = _correct_y_x(y_xmax)
        self.plot(np.array([x1,x2]), np.array([y1, y2]), color = 'blue')

class Optimizer(Plotter):
    def __init__(self, lr=0.01, MAX_ITERS = None):
        self.lr = lr
        self.x0 = None
        self.cur = None
        self.dfx_c = None
        self.f = None
        self.stop_EPS = 1e-3
        self.MAX_ITERS= MAX_ITERS
    def run(self, f, x, x0 = None, ALGORITHM = 'SGD'):
        if x0 is None:
            x0 = np.random.choice(x)
        if self.MAX_ITERS is None:
            self.MAX_ITERS = 0.8*x.shape[0]
        self.x0 = x0
        self.x_c = x0
        self.set_data(x, f)
        self.dfx_c , dfx2= self.diff(self.f, self.x_c)
        fig = plt.figure()
        plt.title(f"Learning rate: {self.lr}")
        cond = True
        k = 1
        writer = PillowWriter(fps = 5)
        with writer.saving(fig, f"output/plot{self.lr}.gif",self.MAX_ITERS):
            while cond and k<self.MAX_ITERS:
                # fig = plt.figure()
                self.plot_iter()
                plt.title(f"Learning rate: {self.lr}")
                writer.grab_frame()
                plt.clf()
                
                cond = self.step()
                k+=1

    def step(self):
        prev = self.x_c
        self.x_c = self.x_c-self.lr*self.dfx_c
        self.dfx_c, dfx2 = self.diff(self.f, self.x_c)
        return np.abs(self.dfx_c)>self.stop_EPS
    def diff(self, f, x0):
        EPS = 1e-6
        cmp = f(x0+EPS*1j)
        return np.imag(cmp)/(EPS), 2*(f(x0)-np.real(cmp))/(EPS**2)
    def plot_iter(self):
        self.plot(self.x, self.f(self.x), color='green')
        xc = self.x_c
        fxc = self.f(xc)
        dfxc = self.dfx_c
        # print("CURRENT", xc, fxc, dfxc)
        self.scatter(xc, fxc, color = 'orange')
        self.draw_line(xc, fxc, dfxc)

    


    

