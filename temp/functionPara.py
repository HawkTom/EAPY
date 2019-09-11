#!/usr/bin/env python
"""
configuration for continue function parameters
"""

__authors__ = "Hao Tong"
__date__ = "2018/03/26"
__version__ = "0.0.1"

import numpy as np


class ContinueFunctionParameter:

    def __init__(functionName, functionBound, functionDimension = 10, functionInitialBound = None, '''
                 ''' target = 'min', opitmalX = None, opitmalY = None): 
        if not functionInitialBound:
            functionInitialBound = functionBound
        if not opitmalX:
            opitmalX = float('inf') * np.ones((1, functionDimension), Float)
        if not opitmalY:
            opitmalY = float('inf')
        self.functionName = functionName
        self.target = target
        self.functionDimension = functionDimension
        self.functionBound = updateBound(functionBound, functionDimension)
        self.functionInitialBound = update(functionInitialBound, functionDimension)
        self.opitmalX = opitmalX
        self.opitmalY = opitmalY

    def updateBound(Bound, Dimension):
        if isinstance(Bound, np.ndarray):
            if np.shape(Bounds) == (1, 2): 
                # the input bounds have to be a matrix (1-by-2) not a vector
                Bound = np.kron(np.ones((Dimension, 1)), Bound).conj().transpose()
        else:
            Bound = np.array([[-1], [1]]) * np.ones((2, Dimension)) * Bound
        return Bound

        
