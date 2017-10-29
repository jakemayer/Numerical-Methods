# Two-dimensional interpolation using nearest-neighbor and bilinear interpolation

import numpy as np
import matplotlib.pyplot as plt

# Performs 2D interpolation using given support points and method
# Inputs
#	xarr: x support points
#	yarr: y support points
#	fmap: values of f at all pairs of (xarr, yarr)
#	xpos: x positions to evaluate interpolation
#	ypos: y positions to evaluate interpolation
#	iord: 0: nearest-neighbor, 1: bilinear interpolation
# Returns
#	fint: 2D array of interpolated function values
def interpol2d(xarr,yarr,fmap,xpos,ypos,iord):

    fint = [[0 for m in range(len(ypos))] for n in range(len(xpos))]
    xCtr = 0
    yCtr = 0
    xint = 0
    yint = 0
    
    for i in range(len(xpos)):
        
        x = xpos[i]
        while (x > xarr[xCtr + 1]):
            xCtr += 1
        
        lDist = x - xarr[xCtr]
        rDist = xarr[xCtr + 1] - x
        
        yCtr = 0        
        for j in range(len(ypos)):
            
            y = ypos[j]
            while (y > yarr[yCtr + 1]):
                yCtr += 1
            
            tDist = y - yarr[yCtr]
            bDist = yarr[yCtr + 1] - y
            
            if (iord == 0):
                xint = xCtr if lDist < rDist else xCtr + 1
                yint = yCtr if tDist < bDist else yCtr + 1
                fint[i][j] = fmap[xint][yint]
            elif (iord == 1):
                xint1 = (rDist*fmap[xCtr][yCtr] + lDist*fmap[xCtr+1][yCtr])/(lDist + rDist)
                xint2 = (rDist*fmap[xCtr][yCtr+1] + lDist*fmap[xCtr+1][yCtr+1])/(lDist + rDist)
                fint[i][j] = (bDist*xint1 + tDist*xint2)/(tDist + bDist)
  
    return(fint)

# Example function
def ff(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.cos(r)**2

def main():
    nres = [4, 8, 16, 32, 64, 128]  # The grid resolutions
    nrand= 100  # The number of random points along x and y to be interpolated (total of 10000 points)
    
    rmse0L = []
    rmse1L = []
    
    # Extent of grid 
    pi = 3.14159
    xmin = -0.5*pi  
    xmax =  0.5*pi
    ymin = -0.5*pi
    ymax =  0.5*pi

    niter  = len(nres)   # Number of resolution steps
    
    # Generate random positions in grid
    xpos = (xmax - xmin)*np.random.rand(nrand) + xmin  # n random positions for x
    ypos = (ymax - ymin)*np.random.rand(nrand) + ymin  # n random positions for y
    xpos.sort()
    ypos.sort()
    xpos[0] = xmin
    xpos[nrand - 1] = xmax
    ypos[0] = ymin
    ypos[nrand - 1] = ymax
    fpos = [[ff(i, j) for j in ypos] for i in xpos]  # The n x n true function values, needed to calculate RMSE

    for n in nres:

        xarr = np.linspace(xmin, xmax, n) # Support points in x direction
        yarr = np.linspace(ymin, ymax, n) # Support points in y direction
        fsup = [[ff(i, j) for j in yarr] for i in xarr] # Function evaluated at support points

        # Call interpolation routines
        fint1 = interpol2d(xarr, yarr, fsup, xpos, ypos, 0)  # The interpolated function values: 0 for nearest
        fint2 = interpol2d(xarr, yarr, fsup, xpos, ypos, 1)  # The interpolated function values: 1 for linear

        # Calculation of RMSEs
        rmse0 = 0
        rmse1 = 0
        for i in range(len(xpos)):
            for j in range(len(ypos)):
                rmse0 += (fint1[i][j] - fpos[i][j])**2
                rmse1 += (fint2[i][j] - fpos[i][j])**2
        rmse0 /= len(xpos)*len(ypos)
        rmse1 /= len(xpos)*len(ypos)
        rmse0 = rmse0**(.5)
        rmse1 = rmse1**(.5)
        
        rmse0L.append(rmse0)
        rmse1L.append(rmse1)
    
    # Log-log plot of RMSE by resolution
    m0, b0 = np.polyfit(np.log(nres), np.log(rmse0L), 1)
    m1, b1 = np.polyfit(np.log(nres), np.log(rmse1L), 1)
        
    plt.scatter(np.log(nres), np.log(rmse0L), c='r', label='nearest neighbor')
    plt.scatter(np.log(nres), np.log(rmse1L), c='g', label='bilinear interpolation')
    plt.plot(np.log(nres), m0*np.log(nres) + b0, c='r')
    plt.plot(np.log(nres), m1*np.log(nres) + b1, c='g')
    
    plt.title('Error vs. Resolution for Two Interpolation Techniques')
    plt.xlabel('log(nres)')
    plt.ylabel('log(rmse)')
    plt.legend(loc='lower left')
    plt.show()
    
    print('Slope for nearest neighbor: ' + str(m0))
    print('Slope for bilinear interpolation: ' + str(m1))
    
main()