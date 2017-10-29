# Solving of differential equations using fixed-step integration

import numpy as np
import matplotlib.pyplot as plt

# Advances solution of f at (x, y) by Euler step of length h
def eulerstep(y,f,x,h):
    return y + h*f(x, y)

# Advances solution of f at (x, y) by Runge-Kutta 2nd-order step of length h
def rk2step(y,f,x,h)
    k1 = f(x, y)
    k2 = f(x + (1/2)*h, y + (1/2)*h*k1)

    return y + h*k2

# Advances solution of f at (x, y) by Runge-Kutta 4th-order step of length h
def rk4step(y,f,x,h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + (h/2)*k1)
    k3 = f(x + h/2, y + (h/2)*k2)
    k4 = f(x + h, y + h*k3)
    
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# Advances solution of f from x0 to x1 using the given step method and number of steps
def ode_fixedstep(nstep,f,x0,y0,x1,iinteg):
    h = (x1 - x0)/nstep
    x = np.linspace(x0, x1, nstep + 1)
    y = [y0]
    
    if (iinteg == 0):
        stepFunc = eulerstep
    elif (iinteg == 1):
        stepFunc = rk2step
    elif (iinteg == 2):
        stepFunc = rk4step
    
    for i in range(nstep):
        ynew = stepFunc(y[i], f, x[i], h)
        y.append(ynew)

    return (x, y)

# Example function
def f(x,y):
    return -2.0*x - y

# Analytical solution to example function
def solution(x):
    return -3.0*np.exp(-x) - 2.0*x + 2

def main():
    
    nstep  = [10, 100, 1000, 10000];   # number of steps
    iinteg = [0, 1, 2];    # type of integrator
    x0     = 0.0;  # starting x
    y0     = -1.0; # starting y
    x1     = 3.0;  # end x

    rmseData = np.zeros((3, 4))
    stepNames = ["Euler", "RK2", "RK4"]
    colors = ['r', 'g', 'b', 'y']
    
    for i in range(len(iinteg)):
        for n in range(len(nstep)):
            xy = ode_fixedstep(nstep[n], f, x0, y0, x1, iinteg[i])
            x = xy[0]
            y_ode = xy[1]
            y_true = solution(x)
    
            pointErrors = (y_ode - y_true)/y_true
            rmseData[i, n] = np.sqrt(sum(pointErrors**2)/nstep[n])
    
            plt.plot(x, y_ode, c=colors[n], label=str(nstep[n]) + " Steps")
            if (n == 3):
                plt.plot(x, y_true, c='k', label="Actual")

        # Plot approximations of function over analytical solution
        plt.title("Approximation using " + stepNames[i] + " Step")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
                
    # Show RMSEs for each configuration
    print("RMS Errors")
    print("Type" + 5*' ' + "10 Steps" + 20*' ' + "100 Steps" + 19*' ' + "1000 Steps" + 18*' ' + "10000 Steps")
    for i in range(len(iinteg)):
        printStr = "{0:5}".format(stepNames[i])
        for n in range(len(nstep)):
            printStr += "{0:28.22f}".format(rmseData[i, n])
        print(printStr)

main()