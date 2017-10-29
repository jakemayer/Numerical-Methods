# Operations of triangular systems

import numpy as np

# Finds the inverse of a triangular matrix (upper or lower)
# Inputs
#	M: matrix to be inverted
# Returns
#	Minv: the inverted matrix
def triangleInverse(M):

    upperOrLower = 0
    for i in range(M.shape[0]):
        for j in range(i):
            if (M[i, j] != 0):
                upperOrLower = 1
    
    Minv = np.zeros(M.shape)
    
    for i in range(M.shape[0]):
        b = np.zeros(M.shape[0])
        b[i] = 1
        col = triSolve(M, b)
        for j in range(M.shape[0]):
            Minv[j, i] = col[j]
    
    return Minv

# Multiplies a matrix M by a vector v
# Inputs
#	M: matrix to multiply
#	v: vector to multiply
# Returns
#	b: the product vector
def multiplyMatrixVector(M, v):
    b = np.zeros(M.shape[0])
    
    if (M.shape[1] != v.size):
        return "Error: Invalid dimensions"
    
    for i in range(M.shape[0]):
        rowSum = 0
        for j in range(M.shape[1]):
            rowSum += M[i, j]*v[j]
        b[i] = rowSum
    
    return b

# Solves the triangular system Mx = b
# Inputs
#	M: triangular matrix in product
#	b: product vector
# Returns
#	x: the solution to the system
def triSolve(M, b):
    
	upperOrLower = 0
    for i in range(M.shape[0]):
        for j in range(i):
            if (M[i, j] != 0):
                upperOrLower = 1

    x = np.zeros(b.size)
    
    if (upperOrLower == 0):
        for i in range(b.size - 1, -1, -1):
            rowSum = b[i]
            for j in range(b.size - 1, i, -1):
                rowSum -= M[i, j]*x[j] 
            x[i] = rowSum / M[i, i]
    elif (upperOrLower == 1):
        for i in range(b.size):
            rowSum = b[i]
            for j in range(i):
                rowSum -= M[i, j]*x[j] 
            x[i] = rowSum / M[i, i]
    else:
        return "Invalid value for upperOrLower"
        
    return x

def main():

	# Demo of operations
    A = np.array([[9, 0, 0],
                  [-4, 2, 0],
                  [1, 0, 5]])
    Ainv = triangleInverse(A)
    
    b1 = np.array([1, 2, 3])
    b2 = np.array([1, 1, 1])
    b3 = np.array([7, 2, 8])
    
    x1t = triSolve(A, b1)
    x2t = triSolve(A, b2)
    x3t = triSolve(A, b3)
    
    x1i = multiplyMatrixVector(Ainv, b1)
    x2i = multiplyMatrixVector(Ainv, b2)
    x3i = multiplyMatrixVector(Ainv, b3)
    
    print("x1 from triSolve: " + str(x1t) + "\nx1 from inverse: " + str(x1i))
    print("\nx2 from triSolve: " + str(x2t) + "\nx2 from inverse: " + str(x2i))
    print("\nx3 from triSolve: " + str(x3t) + "\nx3 from inverse: " + str(x3i))
    
main()