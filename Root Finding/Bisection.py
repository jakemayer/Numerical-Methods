# One dimensional root finding using bisection

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
#	x: location of root
#	i: number of iterations used
def rf_bisect(f, x1, x2, tol, maxiter):
    if (f(x1)*f(x2) > 0):
        print("No detectable roots in given interval")
        return
    
    for i in range(1, maxiter + 1):
        x = (x1 + x2)/2
        if (f(x)*f(x1) > 0):
            x1 = x
        else:
            x2 = x
        if (np.abs(f(x)) <= tol):
            return (x, i)
    
    return (x, i)

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
    x = np.arange(0,1,.01)
    plt.plot(x, f1(x), label = 'f1', color = 'r')
    plt.plot(x, f2(x), label = 'f2', color = 'g')
    plt.plot(x, f3(x), label = 'f3', color = 'b')
    plt.legend(loc = 'lower right')
    plt.show()
    
    for i in range(1,4):
        print("Root of f1 bracketed in domain [0,1], tol=1e-" + str(3*i) + ", maxiter = 25: " + str(rf_bisect(f1, 0., 1., 10**-(3*i), 25)))
        print("Root of f2 bracketed in domain [0,1], tol=1e-" + str(3*i) + ", maxiter = 25: " + str(rf_bisect(f2, 0., 1., 10**-(3*i), 25)))
        print("Root of f3 bracketed in domain [0,1], tol=1e-" + str(3*i) + ", maxiter = 25: " + str(rf_bisect(f3, 0., 1., 10**-(3*i), 25)) + "\n")
    
main()