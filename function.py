#!/usr/bin/env python
"""
function.py

Description:    some continue benchmark function 

Benchmark:

    Sphere function:        f(\textbf{x}) = \sum_{i=1}^{n} x_i^{2}

    Rastrigin function:     f(\textbf{x}) = \sum_{i=1}^{n}(x_i^2 - 10cos(2\pi x_i) + 10)

    Griewank function:      f(\textbf{x}) = 1 + \sum_{i=1}^{n} \frac{x_i^{2}}{4000} - \prod_{i=1}^{n}cos(\frac{x_i}{\sqrt{i}})

    Rosenbrock function:    f(\textbf{x}) = \sum_{i=1}^{n}[b (x_{i+1} - x_i^2)^ 2 + (a - x_i)^2]

    Schwefel12 function:    f(\textbf{x}) = \sum_{i=1}^{n}\left(\sum_{j=1}^{i}x_j\right)^2

Input: 

    input have to be an numupy matrix, the vector objector will raise an error
    e.g. np.array([[1, 2, 3]]),  np.array([1, 2, 3][np.newaixs, :])

Output:

    number for one solution input
    numpy vector for batch input

Example:

    import continueFunction as cF
    cF.Sphere(np.array([[1, 2, 3]]))
    cF.Rastrigin(np.array([[1, 2, 3], [4, 5, 6]]))

"""

import numpy as np


__all__ = ['continueFunction']

class continueFunction():

    @staticmethod
    def sphere(x):
        bias = -450.0

        offsets = [-39.3119, 58.8999, -46.3224, -74.6515, -16.7997, -80.5441,
               -10.5935, 24.9694, 89.8384, 9.1119, -10.7443, -27.8558, -12.5806,
               7.593, 74.8127, 68.4959, -53.4293, 78.8544, -68.5957, 63.7432,
               31.347, -37.5016, 33.8929, -88.8045, -78.7719, -66.4944, 44.1972,
               18.3836, 26.5212, 84.4723, 39.1769, -61.4863, -25.6038, -81.1829,
               58.6958, -30.8386, -72.6725, 89.9257, -15.1934, -4.3337, 5.343,
               10.5603, -77.7268, 52.0859, 40.3944, 88.3328, -55.8306, 1.3181,
               36.025, -69.9271, -8.6279, -56.8944, 85.1296, 17.6736, 6.1529,
               -17.6957, -58.9537, 30.3564, 15.9207, -18.0082, 80.6411,
               -42.3912, 76.2776, -50.1652, -73.5736, 28.3369, -57.9905,
               -22.7327, 52.0269, 39.2599, 10.8679, 77.8207, 66.0395, -50.0667,
               55.7063, 73.7141, 38.5296, -56.7865, -89.6477, 37.9576, 29.472,
               -35.4641, -31.7868, 77.3235, 54.7906, -48.2794, 74.2714, 72.6103,
               62.964, -14.1446, 20.4923, 46.5897, -83.6021, -46.4809, 83.7373,
               -79.6611, 24.3479, -17.2303, 72.3404, -36.4022]
        
        num_variables = x.shape[1]
        offset = offsets[0: num_variables]
        z = x - np.array(offset)[np.newaxis, :]
        out = np.sum(np.power(z, 2), axis=1) + bias
        # print(out.shape)
        return out[0] if out.size == 1 else out

    @staticmethod
    def Sphere(X):
        out = np.sum(np.power(X, 2), axis=1)
        return out[0] if out.size == 1 else out

    @staticmethod
    def Rastrigin(X):
        out = np.sum(np.power(X, 2) - 10.0*np.cos(2.0 * np.pi * X) + 10, axis=1)
        return out[0] if out.size == 1 else out

    @staticmethod
    def Griewank(X):
        # X.shape[0]: population size   X.shape[1]: dimension
        temp = np.kron(np.ones((X.shape[0], 1)), 1 + np.array(range(X.shape[1])))
        out = 1 + np.sum(np.power(X, 2), axis=1) / 4000 - np.prod(np.cos(X/np.sqrt(temp)), axis=1) 
        return out[0] if out.size == 1 else out

    @staticmethod
    def Rosenbrock(X):
        # X.shape[0]: population size   X.shape[1]: dimension
        out = np.sum(100 * np.power((np.power(X[:, 0:-1], 2) - X[:, 1:]), 2) + np.power((X[:, 0:-1] - 1), 2), axis=1)
        return out[0] if out.size == 1 else out

    @staticmethod
    def Schwefel12(X):
        size = X.shape
        y = np.zeros((size[0], ))
        for i in range(size[1]):            
            y += np.power(np.sum(X[:, 0: i+1], axis=1), 2)
        return y[0] if y.size == 1 else y


