#!/usr/bin/env python
"""
configuration for continue function parameters
"""

__authors__ = "Hao Tong"
__date__ = "2018/03/26"
__version__ = "0.0.1"

import numpy as np
import datetime


class ContinueFunctionParameter:

    def __init__(self, functionName, functionBound, functionDimension=10, functionInitialBound=None, target='min', opitmalX=None, opitmalY=None):
        if not functionInitialBound:
            functionInitialBound = functionBound
        if not opitmalX:
            opitmalX = float('inf') * np.ones((1, functionDimension), float)
        if not opitmalY:
            opitmalY = float('inf')
        self.functionName = functionName
        self.target = target
        self.functionDimension = functionDimension
        self.functionBound = self.updateBound(functionBound, functionDimension)
        self.functionInitialBound = self.updateBound(
            functionInitialBound, functionDimension)
        self.opitmalX = opitmalX
        self.opitmalY = opitmalY

    def updateBound(self, Bound, Dimension):
        if isinstance(Bound, np.ndarray):
            if np.shape(Bounds) == (1, 2):
                # the input bounds have to be a matrix (1-by-2) not a vector
                Bound = np.kron(np.ones((Dimension, 1)),
                                Bound).conj().transpose()
        else:
            Bound = np.array([[-1], [1]]) * np.ones((2, Dimension)) * Bound
        return Bound


class TestParameter:

    def __init__(self, testNumber=30, testLog=True, testSeed=int(datetime.date.today().strftime("%Y%m%d"))):
        if not isinstance(testNumber, int):
            raise NameError("test number format need to be int")
        if not isinstance(testLog, bool):
            raise NameError("test seed format need to be boolean")
        if not isinstance(testLog, (int, float)):
            raise NameError("test seed format need to be numerical")
        self.testNumber = testNumber
        self.testLog = testLog
        self.testSeed = testSeed
        self.testSeeds = testSeed + 2 * np.array(range(testNumber))


class AlgorithmParameter:

    def __init__(self, FEMax=30000, populationSize=10):
        self.algorithmName = "CLPSO"
        self.FEMax = FEMax
        self.populationSize = populationSize
        self.iterationMax = np.ceil(FEMax / populationSize)
        self.initialSeed = 0
