# Performing the forward and backward Fourier transforms

import numpy as np
import matplotlib.pyplot as plt

# Performs a forward or backward Fourier transform on the given functions
# Inputs
#	realIn: real part of the function to be transformed
#	imagIn: imaginary part of the function to be transformed
#	direction: forward = 1, backward = -1
def sdft(realIn, imagIn, direction):
    
    N = realIn.size
    realOut = np.zeros(N)
    imagOut = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            exp = 2*direction*np.pi*i*j/N
            realOut[i] += realIn[j]*np.cos(exp) + imagIn[j]*np.sin(exp)
            imagOut[i] += imagIn[j]*np.cos(exp) - realIn[j]*np.sin(exp)
    
    if (direction == -1):
        realOut /= N
        imagOut /= N
    
    return (realOut, imagOut)

def main():
    
    forward = 1
    backward = -1
    n = 2
    N = 128
    pi = 3.14159265
    
    # Example function
    def f(j):
        return np.sin(2*pi*n*j/N)
    
    j = np.arange(0, N)
    f_real = f(j)
    f_imag = np.zeros(j.size)
    
    ft = sdft(f_real, f_imag, forward)
    ft_real = ft[0]
    ft_imag = ft[1]
    
    ift = sdft(ft_real, ft_imag, backward)
    ift_real = ift[0]
    ift_imag = ift[1]
    
    plt.title("Function before and after two transforms")
    plt.xlabel("j")
    plt.ylabel("f(j)")
    plt.plot(j, f_real, c='r', label='Before')
    plt.plot(j, ift_real, c='b', label='After')
    plt.legend(loc='best')
    plt.show()
    
    plt.title("DFT separated by real and imaginary parts")
    plt.xlabel("k")
    plt.ylabel("F(k)")
    plt.plot(j, ft_real, c='r', label='Real')
    plt.plot(j, ft_imag, c='b', label='Imag')
    plt.legend(loc='best')
    plt.show()

main()