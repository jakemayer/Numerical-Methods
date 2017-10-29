# Solving of matrix systems using Gaussian elimination

import numpy as np

# Solves the system Mx = b via Gaussian Elimination
# Inputs
#	M: matrix in the system
#	b: product vector
# Returns
#	x: solution vector of the system
def gausselim(M, b):
    
    A = M.astype(float)
    x = b.astype(float)
    
    for i in range(x.size):
        if (A[i, i] == 0):
            return "Error: zero pivot"
        x[i] /= A[i, i]
        for j in range(x.size - 1, i - 1, -1):
            A[i, j] /= A[i, i]
        for j in range(x.size):
            if (i != j):
                x[j] -= x[i] * A[j, i]
                for k in range(x.size - 1, -1, -1):
                    A[j, k] -= A[i, k] * A[j, i]
    
    return x

def testCase(A, b, number):
    x = gausselim(A, b)
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