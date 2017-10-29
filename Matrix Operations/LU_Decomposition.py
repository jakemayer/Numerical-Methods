# Solving of matrix systems using LU decomposition

import numpy as np

# Performs LU decomposition on a matrix M
# Inputs
#	M: matrix to decompose
# Returns
#	LU: lower and upper matricies packed together
def ludecomp(M):

    LU = M.astype(float)
    
    for i in range(LU.shape[0]):
        if (LU[i, i] == 0):
            return "Error: zero pivot"
        for j in range(i + 1, LU.shape[1]):
            for k in range(LU.shape[1] - 1, i, -1):
                LU[j, k] -= LU[i, k] * (LU[j, i] / LU[i, i])
            LU[j, i] = LU[j, i] / LU[i, i]
    
    return LU

# Uses an LU decomposition to solve the system LUx = b
# Inputs
#	L: lower matrix
#	U: upper matrix
#	b: product vector
# Returns
#	x: solution vector
def lusolve(L,U,b):
    
    x = np.zeros(b.size)
    y = np.zeros(b.size)
    
    for i in range(b.size):
        rowSum = b[i]
        for j in range(i):
            rowSum -= L[i, j]*y[j]
        y[i] = rowSum / L[i, i]
        
    for i in range(y.size - 1, -1, -1):
        rowSum = y[i]
        for j in range(y.size - 1, i, -1):
            rowSum -= U[i, j]*x[j]
        x[i] = rowSum / U[i, i]
    
    return x

# Split a packed LU matrix into individual L and U matricies
# Inputs
#	LU: packed LU matrix
# Returns
#	L: lower matrix
#	U: upper matrix
def splitLU(LU):
    L = np.zeros(LU.shape)
    U = LU
    
    for i in range(L.shape[0]):
        for j in range(i):
            L[i, j] = LU[i, j]
            U[i, j] = 0
        L[i, i] = 1
    
    return (L, U)

def testCase(A, b, number):
    LU = ludecomp(A)
    LandU = splitLU(LU)
    L = LandU[0]
    U = LandU[1]
    x = lusolve(L, U, b)
    print("x" + number + ": " + str(x))

def main():
    # Test case 1
    A1 = np.array([[4, -2, 1],
                   [-3, -1, 4],
                   [1, -1, 3]])
    b1 = np.array([15, 8, 13])
    testCase(A1, b1, "1")
    
    # Test case 2
    A2 = np.array([[2, 2, 3, 2],
                   [0, 2, 0, 1],
                   [4, -3, 0, 1],
                   [6, 1, -6, -5]])
    b2 = np.array([-2, 0, -7, 6])
    testCase(A2, b2, "2")
    
    # Test case 3
    A3 = np.array([[1.01, 0.99],
                   [0.99, 1.01]])
    b3 = np.array([2.0, 2.0])
    testCase(A3, b3, "3")
    
    # Alternate test case 3
    b3alt = np.array([1.98, 2.02])
    testCase(A3, b3alt, "3alt")

main()