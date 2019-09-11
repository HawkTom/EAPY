#!/usr/bin/env python
"""
functionConfig.py

Description:  setting some parameters for optimizing function

Parameters: 

    funcName:               the name of optimizing function
    funDim:                 the dimension of optimizating function
    funcLowerBound:         the lower bound of search space
    funcUpperBound:         the upper bound of search space
    funcInitLowerBound:     the lower bound for initial 
    funcInitUpperBound:     the upper bound for initail
    target:                 the target for opitmizatio, 'min'(defalut) or 'max' 
    opitmalX:               known optimal solution, none defalut
    optimalY:               known optimal value, none default

PS: 1.  the bound has two input form: a number or a range in list
        a number means all dimension has the same seaech space
        a list means each dimension has different search space
        e.g. funcLowerBound = -10,  funcLowerBound = [-10, -20, -30]
    
    2. if the initial bound is same as search space, no need to input


Eample:

    from functionConfig import *
    cfp1 = ContinueFunctionParameter("Sphere", 10, -10, 10)
    cfp2 = ContinueFunctionParameter("Sphere", 10, -10, 10, -10, 5)
    cfp3 = ContinueFunctionParameter("Sphere", 10, [-10, -20], 10, -10, 5)

"""

import numpy as np
import datetime

__all__ = ['ContinueFunctionParameter']


class ContinueFunctionParameter:

    def __init__(self,
                 funcName,
                 funcDim,
                 funcLowerBound,
                 funcUpperBound,
                 funcInitLowerBound=None,
                 funcInitUpperBound=None,
                 target='min',
                 opitmalX=None,
                 opitmalY=None):

        funcLowerBound = np.array(funcLowerBound)
        funcUpperBound = np.array(funcUpperBound)
        if not funcInitLowerBound:
            funcInitLowerBound = funcLowerBound
        else:
            funcInitLowerBound = np.array(funcInitLowerBound)
        if not funcInitUpperBound:
            funcInitUpperBound = funcUpperBound
        else:
            funcInitUpperBound = np.array(funcInitUpperBound)
        assert funcDim > 0, 'the function dimension must be greater than 0'
        assert (funcLowerBound <= funcUpperBound).all(
        ), 'search lower bounds must less than the upper bounds'
        assert (funcInitLowerBound <= funcInitUpperBound).all(
        ), 'initial search lower bounds must less than the upper bounds'

        if not opitmalX:
            opitmalX = float('inf') * np.ones((1, funcDim), float)
        if not opitmalY:
            opitmalY = float('inf')

        self.funcName = funcName
        self.funcDim = funcDim
        self.target = target
        self.funcLowerBound = self.__updateBound(funcLowerBound, funcDim)
        self.funcUpperBound = self.__updateBound(funcUpperBound, funcDim)

        self.funcInitLowerBound = self.__updateBound(
            funcInitLowerBound, funcDim)
        self.funcInitUpperBound = self.__updateBound(
            funcInitUpperBound, funcDim)

        self.opitmalX = opitmalX
        self.opitmalY = opitmalY

    def __updateBound(self, Bound, Dimension):
        if Bound.size == 1:
            Bound = np.kron(np.ones((Dimension, )), Bound)
        return Bound
