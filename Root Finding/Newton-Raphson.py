# Two-dimensional root finding using Newton-Raphson / fixed-point iteration

import numpy as np

# Finds root of g using 2D Newton-Raphson method
# Inputs
#	g1: 2D function representing real part of g
#	g1x: partial derivative of g1 by x
#	g1y: partial derivative of g1 by y
#	g2: 2D function representing imaginary part of g
#	g2x: partial derivative of g2 by x
#	g2y: partial derivative of g2 by y
#	x0: starting position of x
#	y0: starting position of y
#	tol: tolerance of value considered to be a root
#	maxiter: max number of iterations before stopping
# Returns
#	x0: x coordinate of root
#	y0: y coordinate of root
#	i: number of iterations used
def rf_newton2D(g1, g1x, g1y, g2, g2x, g2y, x0, y0, tol, maxiter):
    
    def det(x,y):
        return g1x(x,y)*g2y(x,y) - g1y(x,y)*g2x(x,y)
    
    def deltaX(x,y):
        return -(g2y(x,y)*g1(x,y) - g1y(x,y)*g2(x,y))/det(x,y)
    
    def deltaY(x,y):
        return -(g1x(x,y)*g2(x,y) - g2x(x,y)*g1(x,y))/det(x,y)
    
    for i in range(0, 1 + maxiter):
        if (np.abs(g1(x0, y0) <= tol and np.abs(g2(x0, y0)) <= tol)):
            return (x0, y0, i)
        x0 += deltaX(x0, y0)
        y0 += deltaY(x0, y0)
    
    return (x0, y0, maxiter)

# Example functions for g(z) = z^3 - 1 (g1 is real part, g2 is imaginary part)

def g1(x,y):
    return (x**3) - 3*x*(y**2) - 1
           
def g2(x,y):
    return 3*(x**2)*y - (y**3)
           
def g1x(x,y):
    return 3*(x**2) - 3*(y**2)
    
def g1y(x,y):
    return -6*x*y
    
def g2x(x,y):
    return 6*x*y
    
def g2y(x,y):
    return 3*(x**2) - 3*(y**2)
    
def main():
    x0 = 1.01
    y0 = 1.01
    tol = 1e-3
    maxiter = 25
    
    # Find 3 roots of the function g(z) = z^3 - 1
    print(rf_newton2D(g1, g1x, g1y, g2, g2x, g2y, 1.5, 0.5, tol, maxiter))
    print(rf_newton2D(g1, g1x, g1y, g2, g2x, g2y, -1, 1, tol, maxiter))
    print(rf_newton2D(g1, g1x, g1y, g2, g2x, g2y, -1, -1, tol, maxiter))
    
main()