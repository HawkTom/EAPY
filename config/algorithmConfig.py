#!/usr/bin/env python
"""
algorithmConfig.py

Description:  setting some runing parameters for algorithm

Parameters: 

    algorithmName:  the name of using algorithm 
    FuncEvalMax:    the maximum function evaluation numbers, default = 30000
    populationSize: defalut = 10
    intialSeed:     the seed for random generator
    iterationMax:   ceil(FuncEvalMax/populationSize)

Eample:

    from algorithmConfig import *
    ap1 = AlgorithmParameter("wPSO")
    ap2 = AlgorithmParameter("wPSO", 200000)
    ap3 = AlgorithmParameter("wPSO", 200000, 30)

"""
import numpy as np


__all__ = ['AlgorithmParameter']


class AlgorithmParameter:

    def __init__(self, algorithmName, FuncEvalMax=30000, populationSize=10):
        assert FuncEvalMax > 0, 'the maximum fitness evaluations must be greater than 0'
        assert populationSize > 0, 'the population size must be greater than 0'

        self.algorithmName = algorithmName
        self.FEMax = FuncEvalMax
        self.populationSize = populationSize
        self.iterationMax = np.ceil(FuncEvalMax / populationSize)
        self.initialSeed = 0
