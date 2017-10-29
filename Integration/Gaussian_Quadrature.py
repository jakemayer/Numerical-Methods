# Approximation of integrals using Gaussian quadrature

import numpy as np
import matplotlib.pyplot as plt

# Calculates the integral of f using Gaussian quadrature
# Inputs
#	f: function to be integrated
#	Np: number of integration points
# Returns
#	Integral: approximation of the integral
def gaussHermiteInt(f,Np):

    NpString = str(Np) if Np>=10 else "0" + str(Np)

    # Data provided in separate files
    file = open("GaussHermiteData/HermiteXW" + NpString + ".dat", "r")
    lines = file.readlines()
    file.close()
    
    points = []
    weights = []
    for line in lines:
        split = line.split()
        points.append(float(split[0]))
        weights.append(float(split[1]))
        
    Integral = 0.0
    
    for i in range(len(points)):
        Integral += f(points[i])*weights[i]
    
    return Integral

# Gets the estimates of f at multiple points
# Inputs
#	f: function to be integrated
#	points: points to evaluate at
# Returns
#	estimates: approximations of f at given points
def getEstimates(f, points):
    estimates = []
    for i in points:
        estimates.append(gaussHermiteInt(f, i))
    return estimates

def main():
    
    # Example function
    def f(k):
        def g(x):
            return np.sin(k*x)**2
        return g
    
    points = [2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    colors = ['r', 'y', 'g', 'b', 'k']
    
    plt.title("Estimate vs. # of Points for Gauss-Hermite Quadrature")
    plt.xlabel("# of Points")
    plt.ylabel("Estimate")
    
    for i in range(0, 5):
        k = 6 + .2*i
        estimates = getEstimates(f(k), points)
        plt.plot(points, estimates, c=colors[i], label='k = ' + str(k))
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

main()