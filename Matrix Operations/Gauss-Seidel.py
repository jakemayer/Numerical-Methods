# Solving of matrix systems using Gauss-Seidel iteration

import numpy as np
import matplotlib.pyplot as plt

# Solves the system Mx = b using Gauss-Seidel iteration
# Inputs
#	M: matrix in the system
#	b: product vector
#	tol: tolerance of norm of error vector
# Returns
#	xnew: solution vector
#	iterations: number of iterations used
def gaussseidel(M, b, tol):

    M = M.astype(float)
    b = b.astype(float)
    
    xold = np.zeros(b.size)
    xnew = np.zeros(b.size)
    
    L = np.zeros(M.shape)
    U = np.copy(M)
    
    for i in range(M.shape[0]):
        xnew[i] = b[i]/M[i, i]
        for j in range(i+1):
            L[i, j] = U[i, j]
            U[i, j] = 0
    
    Linv = triangleInverseLower(L)
        
    iterations = 0
    while(np.linalg.norm(xnew - xold) > tol):
        iterations += 1
        xold = np.copy(xnew)
        xnew = np.dot(Linv, b - np.dot(U, xold))
    
    return (xnew, iterations)

# Finds the inverse of a lower-triangular matrix
# Inputs
#	M: matrix to invert
# Returns
#	Minv: inverted matrix
def triangleInverseLower(M):
    Minv = np.zeros(M.shape)
    
    for i in range(M.shape[0]):
        b = np.zeros(M.shape[0])
        b[i] = 1
        col = triSolveLower(M, b)
        for j in range(M.shape[0]):
            Minv[j, i] = col[j]
    
    return Minv

# Solves a lower-triangular matrix system
# Inputs
#	M: matrix in the system
#	b: product vector
# Returns
#	x: solution vector
def triSolveLower(M, b):
    x = np.zeros(b.size)
    
    for i in range(b.size):
        rowSum = b[i]
        for j in range(i):
            rowSum -= M[i, j]*x[j] 
        x[i] = rowSum / M[i, i]
        
    return x

def main():
    tol = 1e-4
    
    # Test Case 1
    A1 = np.array([[1.01, 0.99],
                   [0.99, 1.01]]) 
    b1 = np.array([2, 2])
    x1 = gaussseidel(A1, b1, tol)
    print("x1: " + str(x1[0]) + "\n")
    print("Iterations: " + str(x1[1]))
    
    # Test Case 2
    A2 = np.array([[1.5, 0.5],
                   [0.5, 1.5]]) 
    b2 = np.array([2, 2])
    x2 = gaussseidel(A2, b2, tol)
    print("x2: " + str(x2[0]) + "\n")
    print("Iterations: " + str(x2[1]))
    
main()