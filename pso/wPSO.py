# !/usr/bin/env python
"""
wPSO.py

Description:       the implemention of modified particle swarm optimization

Refrence paper:
Shi, Yuhui, and Russell Eberhart. "A modified particle swarm optimizer." 
Evolutionary Computation Proceedings, 1998. IEEE World Congress on Computational Intelligence.

Member variables:

     Name:                      wPSO
     FERuntime:                 time for fitness evaluation
     FENum:                     number of fitness evaluation
     runtime:                   time for whole algorithm
     optimalX:                  optimal solution for problem
     optimalY:                  optimal value for problem
     convergeCurve:             the procedure of convergence
     convergeCurveInterval:     inverval between two saved points
     w(weight):                 default w_max = 0.9, w_min = 0.4 
     learningRate:              default = 2.0

Member function:
     
     setParameters(weight, learningRate):   setting parameters

     optimize(cfp, ap, printLog):           the main process of optimization

                                            cfp:        config for continue function parameters
                                            ap:         config for algorithm parameters
                                            printLog:   determine whether to print Log after opitmization
                                                        (true default)

Example: 

    agent = wPSO()
    agent.optimize(cfp, ap, printLog=True) # cfp ap need to config at first

"""

from function import continueFunction as cF
import numpy as np
import time
import sys
import copy

__all__ = ['wPSO']


class wPSO:

    def __init__(self):
        self.Name = "wPSO"
        self.FERuntime = 0
        self.FENum = 0
        self.setParameters()

    # setting weight, learning rate of PSO
    def setParameters(self, weight=[0.9, 0.4], learningRate=2.0):
        self.w = weight
        self.learningRate = learningRate

    def optimize(self, cfp, ap, printLog=True):
        runtimeStart = time.clock()
        self.__mainLoop(cfp, ap, printLog)
        self.runtime = time.clock() - runtimeStart

    def __mainLoop(self, cfp, ap, printLog):
        np.random.seed(ap.initialSeed)
        popSize = ap.populationSize
        Dim = cfp.funcDim
        function = getattr(cF, cfp.funcName)
        lowerBoundX = np.kron(np.ones((popSize, 1)), cfp.funcLowerBound)
        upperBoundX = np.kron(np.ones((popSize, 1)), cfp.funcUpperBound)

        lowerInitBoundX = np.kron(
            np.ones((popSize, 1)), cfp.funcInitLowerBound)
        upperInitBoundX = np.kron(
            np.ones((popSize, 1)), cfp.funcInitUpperBound)

        upperBoundV = 0.2 * (upperBoundX - lowerBoundX)
        lowerBoundV = -1 * upperBoundV
        # initial X position and velocity
        X = (upperInitBoundX - lowerInitBoundX) * \
            np.random.random_sample((popSize, Dim)) + lowerInitBoundX
        V = (upperBoundV - lowerBoundV) * \
            np.random.random_sample((popSize, Dim)) + lowerBoundV
        start = time.clock()
        y = function(X)
        self.FERuntime += (time.clock()-start)
        self.FENum += popSize
        # initialize personal best X and y
        personBestX, personBestY = copy.deepcopy(X), copy.deepcopy(y)
        # initialize global best X and y
        gBestX, gBestY = X[np.argmin(y), :], np.min(y)
        self.convergeCurve = [y[0], gBestY]
        maxGen, gen = ap.iterationMax, 0
        while self.FENum < ap.FEMax:
            wk = self.w[0] - (self.w[0] - self.w[1]) * gen / maxGen
            for pi in range(popSize):
                # update and limit V
                V[pi, :] = wk * V[pi, :] + self.learningRate * np.random.random_sample((1, Dim)) * (
                    personBestX[pi, :] - X[pi, :]) + self.learningRate * np.random.random_sample((1, Dim)) * (gBestX - X[pi, :])
                V[pi, :][V[pi, :] < lowerBoundV[pi, :]
                         ] = lowerBoundV[pi, :][V[pi, :] < lowerBoundV[pi, :]]
                V[pi, :][V[pi, :] > upperBoundV[pi, :]
                         ] = upperBoundV[pi, :][V[pi, :] > upperBoundV[pi, :]]
                # update X
                X[pi, :] = X[pi, :] + V[pi, :]
                X[pi, :][X[pi, :] < lowerBoundX[pi, :]
                         ] = lowerBoundX[pi, :][X[pi, :] < lowerBoundX[pi, :]]
                X[pi, :][X[pi, :] > upperBoundX[pi, :]
                         ] = upperBoundX[pi, :][X[pi, :] > upperBoundX[pi, :]]
                # update personal and global best X and y
                start = time.clock()
                y[pi] = function(X[pi, :][np.newaxis, :])
                self.FERuntime += (time.clock() - start)
                self.FENum += 1
                if y[pi] < personBestY[pi]:
                    personBestX[pi, :] = X[pi, :]
                    personBestY[pi] = y[pi]
                    if personBestY[pi] < gBestY:
                        gBestX = personBestX[pi, :]
                        gBestY = personBestY[pi]
                if self.FENum % popSize == 0:
                    self.convergeCurve.append(gBestY)
            gen = gen + 1
            # print('Gen:{0} BestV: {1} \n'.format(self.FENum, gBestY))
        self.optimalX = gBestX
        self.optimalY = gBestY
        self.convergeCurveIntrval = popSize
        if printLog:
            # summary
            print('$--------Result--------$\n')
            print('*Function: {0}\tDimension: {1}\t FEMax: {2}\n'.format(
                cfp.funcName, cfp.funcDim, self.FENum))
            print('Optimal Y  : {0} \n'.format(self.optimalY))
