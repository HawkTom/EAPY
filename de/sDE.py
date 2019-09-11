# !/usr/bin/env python
"""
sDe.py

Description:       the implemention of modified simple DE

Refrence paper:
Storn, Rainer, and Kenneth Price. 
"Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces."

Member variables:

     Name:                      sDE
     FERuntime:                 time for fitness evaluation
     FENum:                     number of fitness evaluation
     runtime:                   time for whole algorithm
     optimalX:                  optimal solution for problem
     optimalY:                  optimal value for problem
     convergeCurve:             the procedure of convergence
     convergeCurveInterval:     inverval between two saved points
     CR:                        crossover constant
     F:                         real constant factor for mutation 


Member function:
     
     setParameters(weight, learningRate):   setting parameters

     optimize(cfp, ap, printLog):           the main process of optimization

                                            cfp:        config for continue function parameters
                                            ap:         config for algorithm parameters
                                            printLog:   determine whether to print Log after opitmization
                                                        (true default)

Example: 

    agent = sDE()
    agent.optimize(cfp, ap, printLog=True) # cfp ap need to config at first

"""

from function import continueFunction as cF
import numpy as np
import time
import sys
import copy

__all__ = ['sDE']


class sDE:

    def __init__(self):
        self.Name = "sDE"
        self.FERuntime = 0
        self.FENum = 0
        self.setParameters()

    # setting F, CR
    def setParameters(self, F=0.5, CR=0.9):
        self.F = F
        self.CR = CR

    def optimize(self, cfp, ap, printLog=True):
        runtimeStart = time.clock()
        self.__mainLoop(cfp, ap, printLog)
        self.runtime = time.clock() - runtimeStart

    def __mainLoop(self, cfp, ap, printLog):
        # np.random.seed(ap.initialSeed)
        popSize = ap.populationSize
        Dim = cfp.funcDim
        function = getattr(cF, cfp.funcName)
        lowerBoundX = np.kron(np.ones((popSize, 1)), cfp.funcLowerBound)
        upperBoundX = np.kron(np.ones((popSize, 1)), cfp.funcUpperBound)

        lowerInitBoundX = np.kron(
            np.ones((popSize, 1)), cfp.funcInitLowerBound)
        upperInitBoundX = np.kron(
            np.ones((popSize, 1)), cfp.funcInitUpperBound)

        # initial X position
        X = (upperInitBoundX - lowerInitBoundX) * \
            np.random.random_sample((popSize, Dim)) + lowerInitBoundX
        start = time.clock()
        y = function(X)
        self.FERuntime += (time.clock()-start)
        self.FENum += popSize
        self.convergeCurve = [y[0]]

        # main loop
        while self.FENum < ap.FEMax:
            Xnew = copy.deepcopy(X)
            for pi in range(popSize):
                ind = X[pi, :]
                trail = np.zeros_like(ind)
                # randomly pick 3 vectors different pi
                r = np.random.randint(popSize, size=(3, ))
                while (r == pi).any():
                    r = np.random.randint(popSize, size=(3, ))
                # mutation                     
                v = X[r[0], :] + self.F * (X[r[1], :] - X[r[2], :])
                # crossover
                jPosition = np.random.randint(Dim)
                Position = np.random.random_sample(size=(Dim, )) < self.CR
                Position[jPosition] = True
                trail[Position] = v[Position]
                trail[~Position] = ind[~Position]
                trail[trail < lowerBoundX[pi, :]] = lowerBoundX[pi, :][trail < lowerBoundX[pi, :]]
                trail[trail > upperBoundX[pi, :]] = upperBoundX[pi, :][trail > upperBoundX[pi, :]]
                start = time.clock()
                fitnessTrail = function(trail[np.newaxis, : ])
                self.FERuntime += (time.clock() - start)
                self.FENum += 1
                if fitnessTrail < y[pi]:
                    Xnew[pi, :] = trail
                    y[pi] = fitnessTrail
            gBestX, gBestY = X[np.argmin(y), :], np.min(y)
            self.convergeCurve.append(gBestY)
            X = copy.deepcopy(Xnew)
            # print(X)
            print('Gen:{0} BestV: {1} '.format(self.FENum, gBestY))
        self.optimalX = gBestX
        self.optimalY = gBestY
        self.convergeCurveIntrval = popSize
        if printLog:
            # summary
            print('$--------Result--------$\n')
            print('*Function: {0}\tDimension: {1}\t FEMax: {2}\n'.format(
                cfp.funcName, cfp.funcDim, self.FENum))
            print('Optimal Y  : {0} \n'.format(self.optimalY))        