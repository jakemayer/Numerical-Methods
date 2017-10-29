# Approximation of integrals using Simpson's methods

import numpy as np
import matplotlib.pyplot as plt

# Calculates the integral of f between a and b using Simpson's methods
# Inputs
#	a: lower limit of integration
#	b: upper limit of integration
#	f: function to integrate
#	Np: number of integration points in the approximation
#	method: trapezoidal = 0, Simpson's 1/3 = 1, Simpson's 3/8 = 2
# Returns
#	Integral: approximate value of the integral
def trapezoidInt(a,b,f,Np,method):

    Integral = f(a) - f(b)
    h = (b - a)/Np
    
    if (method == 0):
        for i in range(1, Np + 1):
            Integral += 2*f(a + i*h)
        Integral *= h/2
    if (method == 1):
        for i in range(1, Np + 1):
            Integral += 4*f(a + (i-.5)*h) + 2*f(a + i*h)
        Integral *= h/3
    if (method == 2):
        for i in range(1, Np + 1):
            Integral += 3*f(a + (i-(2/3))*h) + 3*f(a + (i-(1/3))*h) + 2*f(a + i*h)
        Integral *= 3*h/8
    
    return Integral

# Gets the relative error between different resolutions of approximation
def getError(a, b, f, points, method):
    approx = []
    error = []
    leftBoundPts = 10
    approx.append(trapezoidInt(a, b, f, leftBoundPts, method))
    
    for i in range(len(points)):
        approx.append(trapezoidInt(a, b, f, points[i], method))
        error.append(np.abs(approx[i + 1] - approx[i]))
        
    return error

def main():
    
    # Example function
    def f(x):
        return np.log(1 + x)/x if x != 0 else 1
    
    points = [100, 1000, 10000, 100000]
    
    # Get errors for each method
    errorT = getError(0, 1, f, points, 0)
    errorS1 = getError(0, 1, f, points, 1)
    errorS2 = getError(0, 1, f, points, 2)
    
    # Determine slopes of error plots (convergence rate)
    slopeT = np.polyfit(np.log(points), np.log(errorT), 1)[0]
    slopeS1 = np.polyfit(np.log(points)[0:3], np.log(errorS1)[0:3], 1)[0]
    slopeS2 = np.polyfit(np.log(points)[0:3], np.log(errorS2)[0:3], 1)[0]
    print("Slope for Trapezoidal:   " + str(slopeT))
    print("Slope for Simpson\'s 1/3: " + str(slopeS1))
    print("Slope for Simpson\'s 3/8: " + str(slopeS2))
    
    # Plot errors by resolution
    plt.title("Log(error) vs. Log(steps) for 3 Numerical Integrations")
    plt.xlabel("Log(steps)")
    plt.ylabel("Log(error)")
    plt.scatter(np.log(points), np.log(errorT), c='r', label='Trapezoidal')
    plt.scatter(np.log(points), np.log(errorS1), c='g', label='Simpson\'s 1/3')
    plt.scatter(np.log(points), np.log(errorS2), c='b', label='Simpson\'s 3/8')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
main()