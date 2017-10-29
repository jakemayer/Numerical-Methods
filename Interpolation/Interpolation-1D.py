# One-dimensional interpolation using splines and Lagrange polynomials

import numpy as np
import scipy as sp
from scipy import interpolate as spi
import matplotlib.pyplot as plt

# Returns i-th Lagrange function evaulated at x using n support points given in xsup
# Inputs
#	x: the point at which to evaluate the function
#	xsup: the support points used for interpolation
#	i: the degree of the Lagrange function
#	n: the number of support points
# Returns
#	y: the value of the function at x
def LagrangeL(x,xsup,i,n):
    y = 1

    for j in range(0, n):
        if (j != i):
            y *= (x - xsup[j])/(xsup[j] - xsup[i])
    
    return y

# Returns interpolation of function evaluated at x using n support points given in (xsup, ysup)
# Inputs
#	x: the point at which to evaluate the function
#	xsup: the support points used for interpolation
#	ysup: the function evaluated at the points in xsup
#	n: the number of support points
# Returns
#	y: the value of the interpolation at x
def LagrangePoly(x,xsup,ysup,n):
    y = 0
    
    for j in range(0, n):
        y += ysup[j]*LagrangeL(x,xsup,j,n)
    
    return y

# Example function
def fx(x):
    return np.abs(np.sin(x))

def main():
    # Interval of interest
    xmin = -2.0
    xmax = 2.0
    
    rmseLL = []
    rmseSL = []
    
    # Number of support points
    nsup = [3, 5, 7, 9]
    
    # Set to 0 for fixed-distance support points and 1 for random support points
    random = 0 
    
    for n in nsup:
        
        if (random == 0):
            xpos = np.linspace(xmin, xmax, num = n)
        elif (random == 1):
            xpos = (xmax - xmin)*np.random.rand(n) + xmin
            xpos.sort()
            xpos[0] = xmin
            xpos[n - 1] = xmax
        
        # Function values at support points xpos
        ypos = fx(xpos)

        # Number of points in interval, to be evaluated with interpolation
        nres = 1000
    
    	# Interpolation points
        xarr = np.linspace(xmin,xmax,nres) 

        # True function values at interpolation points (for error estimates)
        yarr = fx(xarr)
    
    	# The interpolation result using Lagrange polynomials of degree n
        ypolL = LagrangePoly(xarr, xpos, ypos, n)
        if (n > 3):
            ypolS = spi.interp1d(xpos, ypos, kind = 'cubic')(xarr)  # The interpolation result using cubic Splines

        # The RMSE for each interpolation
        rmseL = np.sqrt(sum((ypolL - yarr)**2)/len(yarr))
        rmseLL.append(rmseL)

        if (n > 3):
            rmseS = np.sqrt(sum((ypolS - yarr)**2)/len(yarr))
            rmseSL.append(rmseS)
        else:
            rmseSL.append('N/A')
    
        # Plot of all interpolations and support points
    
        plt.plot(xarr, fx(xarr), color='r', label='Function')
        
        if (n > 3):
            plt.plot(xarr, ypolS, color='g', label='Spline')
        
        plt.plot(xarr, ypolL, color='b', label='Polynomial')
        plt.scatter(xpos, ypos, color='k', label='Support')
        plt.legend(loc='lower right', fontsize='medium')
        plt.title('Interpolation with ' + str(n) + ' Support Points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-2,2)
        plt.show()

    # Table of RMSEs
    print("%-*s%-*s%-*s" % (20, 'Support Points', 20, 'Polynomial RMSE', 20, 'Spline RMSE'))
    for i in range(len(nsup)):
        print("%-*s%-*s%-*s" % (20, nsup[i], 20, rmseLL[i], 20, rmseSL[i]))
        
main()