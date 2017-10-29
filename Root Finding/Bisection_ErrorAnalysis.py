# Visualization and error analysis of bisection method

import numpy as np
import matplotlib.pyplot as plt

# Finds value of root using bisection, within a given tolerance or number of iterations
# Inputs
# 	f: function to perform bisection on
# 	x1: left bound of neighborhood of root
# 	x2: right bound of neighborhood of root
# 	tol: tolerance of value considered to be a root
# 	maxiter: max number of iterations before stopping
# Returns
#	xvals: np array of values of approximated root locations
#	fvals: np array of values of function at xvals locations
def rf_bisect(f, x1, x2, tol, maxiter):
    if (f(x1)*f(x2) > 0):
        print("No detectable roots in given interval")
        return
    
    xVals = []
    fVals = []
    
    for i in range(1, maxiter + 1):
        x = (x1 + x2)/2
        xVals.append(x)
        fVals.append(f(x))
        
        if (f(x)*f(x1) > 0):
            x1 = x
        else:
            x2 = x
        if (np.abs(f(x)) <= tol):
            return (np.array(xVals), np.array(fVals))
    
    return (np.array(xVals), np.array(fVals))

# Plots a given function overlayed by its root approximations, and a plot of error vs. number of iterations
# Inputs
#	f: the functions to be plotted
#	name: the name used to label the graphs
#	clr: the color to plot the graphs in
def plotFunc(f, name, clr):
    x = np.arange(0,1,.01)
    fplot = rf_bisect(f, 0., 1., 0, 25)
    
    plt.scatter(fplot[0], fplot[1], label = name + ' bisection', color = 'k')
    plt.plot(x, f(x), label = name, color = clr)
    
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.xlim(0,1)
    plt.legend(loc = 'lower right')
    plt.show()
    
    x = np.arange(1,25,1)
    plt.scatter(x, fplot[0][x - 1] - fplot[0][24], color = clr)
    plt.xlabel('# of iterations')
    plt.ylabel('Error for ' + name)
    plt.xlim(0,25)
    plt.show()
    
# Example functions

def f1(x):
    return 3 * x + np.sin(x) - np.exp(x)
    
def f2(x):
    return x**3

def f3(x):
    return np.sin(1. / (x + 0.01))

def f4(x):
    return 1. / (x - 0.5)

def main():
    plotFunc(f1, 'f1', 'r')
    plotFunc(f2, 'f2', 'g')
    plotFunc(f3, 'f3', 'b')
    
main()