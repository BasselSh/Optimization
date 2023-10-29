import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from abc import abstractmethod
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
    def __init__(self, lr=0.01, MAX_ITERS = None, write = True, root = "output"):
        self.root = root
        self.write = write
        self.lr = lr
        self.x0 = None
        self.x_c = None
        self.dfx_c = None
        self.f = None
        self.stop_EPS = 1e-3
        self.MAX_ITERS= MAX_ITERS
        self.RESOLUTION = 100

    def _pre_loop(self,f,xl,xr,x0):
        self.xl = xl
        self.xr = xr
        x = np.linspace(xl,xr,self.RESOLUTION)
        self.minstep = (xr-xl)/self.RESOLUTION
        if x0 is None:
            x0 = np.random.choice(x)
        if self.MAX_ITERS is None:
            self.MAX_ITERS = 0.8*x.shape[0]
        self.x0 = x0
        self.x_c = x0
        self.set_data(x, f)
        self.dfx_c , self.df2x_c= self.diff(self.f, self.x_c)
    def before_loop(self):
        pass  
    def first_plot(self):
        pass
    def run(self, f, xl, xr, x0 = None, ALGORITHM = 'SGD'):
        self._pre_loop(f, xl, xr, x0)
        self.before_loop()
        cond = True
        k = 1
        self.iter = k
        if self.write:
            fig = plt.figure()
            plt.title(f"Learning rate: {self.lr}")
            self.first_plot()
            writer = PillowWriter(fps = 5)
            with writer.saving(fig, f"{self.root}/plot{self.lr}.gif",self.MAX_ITERS):
                while cond and k<self.MAX_ITERS:
                    self.plot_iter()
                    plt.title(f"Learning rate: {self.lr}")
                    writer.grab_frame()
                    plt.clf()
                    cond = self.step()
                    k+=1
        else:
            while cond and k<self.MAX_ITERS:
                cond = self.step()
                k+=1
        self.plot_after_loop()
        # self.x_star = self.cur
        # self.y_star = self.f(self.x_star)
    def plot_after_loop(self):
        pass
    def diff(self, f, x0):
        EPS = 1e-6
        cmp = f(x0+EPS*1j)
        return np.imag(cmp)/(EPS), 2*(f(x0)-np.real(cmp))/(EPS**2)

    @abstractmethod
    def step(self):
        pass
    @abstractmethod
    def plot_iter(self):
        pass

class SGD(Optimizer):
    def __init__(self, lr=0.01, MAX_ITERS = None, root = "output"):
        super().__init__(lr = lr, MAX_ITERS = MAX_ITERS, root = root)
    def step(self):
        prev = self.x_c
        self.x_c = self.x_c-self.lr*self.dfx_c
        self.dfx_c, dfx2 = self.diff(self.f, self.x_c)
        return np.abs(self.dfx_c)>self.stop_EPS and self.x_c>=self.xl and self.x_c <= self.xr
    def plot_iter(self):
        self.plot(self.x, self.f(self.x), color='green')
        xc = self.x_c
        fxc = self.f(xc)
        dfxc = self.dfx_c
        self.scatter(xc, fxc, color = 'orange')
        self.draw_line(xc, fxc, dfxc)


class LinearSearch(Optimizer):
    def __init__(self):
        pass

class Bracketer(Optimizer):
    def __init__(self, K, lr=0.01, MAX_ITERS = None, root = "output"):
        super().__init__(lr = lr, MAX_ITERS = MAX_ITERS, root = root)
        self.K = K
        self.S = 1
        
    def step(self):
        prev = self.x_c
        self.S *= self.K
        self.x_c = self.x_c + self.sign*self.S
        fprev = self.f(prev)
        fxc = self.f(self.x_c)
        return self.f(self.x_c)<self.f(prev) and self.x_c>=self.xl and self.x_c <= self.xr
    def before_loop(self):
        dfx0, df2x0 = self.diff(self.f, self.x0)
        self.sign = -np.sign(dfx0)
        self.S = self.minstep
    def draw_vertical_line(self,x):
        self.plot(np.array([x,x]), np.array([self.f(x),self.YLMAX]), color= 'blue')
    def plot_iter(self):
        self.plot(self.x, self.f(self.x), color='green')
        self.scatter(self.x0, self.f(self.x0), color = 'red')
        self.draw_vertical_line(self.x0)
        xc = self.x_c
        fxc = self.f(xc)
        dfxc = self.dfx_c
        self.scatter(xc, fxc, color = 'orange')
        self.draw_vertical_line(xc)
    def plot_after_loop(self):
        self.plot_iter()
    
        


