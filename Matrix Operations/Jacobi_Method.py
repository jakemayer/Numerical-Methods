# Solving of matrix systems using Jacobi method

import numpy as np
import matplotlib.pyplot as plt

# Solves the system Mx = b using Jacobi iteration
# Inputs
#	M: matrix in the system
#	b: product vector
#	tol: tolerance of norm of error vector
# Returns
#	xnew: solution vector
#	iterations: number of iterations used
def jacobi(M, b, tol):
    
    M = M.astype(float)
    b = b.astype(float)
    
    xold = np.zeros(b.size)
    xnew = np.zeros(b.size)
    
    Dinv = np.zeros(M.shape[0])
    R = np.copy(M)
    
    for i in range(M.shape[0]):
        Dinv[i] = 1/M[i, i]
        R[i, i] = 0
        xnew[i] = b[i]/M[i, i]
    
    iterations = 0
    while(np.linalg.norm(xnew - xold) > tol):
        iterations += 1
        xold = np.copy(xnew)
        xnew = Dinv*(b - np.dot(R, xold))
    
    return (xnew, iterations)

# Multiplies the matrix B by the vector v
def applymatrix(B, v):

    Bv = np.zeros(B.shape[0])
    
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            Bv[i] += B[i, j]*v[j]

    return Bv

def main():
    tol = 1e-4
    
    # Test case 1
    A1 = np.array([[1.01, 0.99],
                   [0.99, 1.01]]) 
    b1 = np.array([2, 2])
    x1 = jacobi(A1, b1, tol)
    print("x1: " + str(x1[0]) + "\n")
    print("Iterations: " + str(x1[1]))

    # Test case 2
    A2 = np.array([[1.5, 0.5],
                   [0.5, 1.5]]) 
    b2 = np.array([2, 2])
    x2 = jacobi(A2, b2, tol)
    print("x2: " + str(x2[0]) + "\n")
    print("Iterations: " + str(x2[1]))

main()