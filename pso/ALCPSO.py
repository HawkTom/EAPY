# !/usr/bin/env python
"""
ALCPSO.py

Description:      the implemention of ALCPSO

Refrence paper:
Chen, Wei-Neng, et al. "Particle swarm optimization with an aging leader and challengers."
IEEE Transactions on Evolutionary Computation 17.2 (2013): 241-258.

Member variables:

     Name:                      ALCPSO
     FERuntime:                 time for fitness evaluation
     FENum:                     number of fitness evaluation
     runtime:                   time for whole algorithm
     optimalX:                  optimal solution for problem
     optimalY:                  optimal value for problem
     convergeCurve:             the procedure of convergence
     convergeCurveInterval:     inverval between two saved points
     challengingTimesMax:       default = 2
     leaderLifeSpanMax:         default = 60

Member functions:
     
     setParameters(weight, learningRate):   setting parameters

     optimize(cfp, ap, printLog):           the main process of optimization

                                            cfp:        config for continue function parameters
                                            ap:         config for algorithm parameters
                                            printLog:   determine whether to print Log after opitmization
                                                        (true default)

Example: 

    agent = ALCPSO()
    agent.optimize(cfp, ap, printLog=True) # cfp ap need to config at first

"""

from function import continueFunction as cF
import numpy as np
import time
import sys
import copy


class ALCPSO:

    def __init__(self):
        self.Name = "ALCPSO"
        self.FERuntime = 0
        self.FENum = 0
        self.setParameters()

    def setParameters(self, challengingTimesMax=2, leaderLifeSpanMax=60):
        self.challengingTimesMax = challengingTimesMax
        self.leaderLifeSpanMax = leaderLifeSpanMax

    def optimize(self, cfp, ap, printLog=True):
        runtimeStart = time.clock()
        self.mainLoop(cfp, ap, printLog)
        self.runtime = time.clock() - runtimeStart

    def mainLoop(self, cfp, ap, printLog):
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

        upperBoundV = 0.5 * (upperBoundX - lowerBoundX)
        lowerBoundV = -1 * upperBoundV
        # initial X position and velocity
        X = (upperInitBoundX - lowerInitBoundX) * \
            np.random.random_sample((popSize, Dim)) + lowerInitBoundX
        V = np.zeros((popSize, Dim))

        start = time.clock()
        y = function(X)
        self.FERuntime += (time.clock()-start)
        self.FENum += popSize
        # initialize personal best X and y
        personBestX, personBestY = copy.deepcopy(X), copy.deepcopy(y)
        # initialize global best X and y
        gBestX, gBestY = X[np.argmin(y), :], np.min(y)
        self.convergeCurve = [y[0], gBestY]
        # initialize leader
        leaderX, leaderY = copy.deepcopy(gBestX), copy.deepcopy(gBestY)
        leaderAge = 0
        leaderLifeSpan = copy.copy(self.leaderLifeSpanMax)
        while self.FENum < ap.FEMax:
            indicatorGoodLeadPower = False
            indicatorFairLeadPower = False
            indicatorPoorLeadPower = False
            personBestYBackUp = copy.deepcopy(personBestY)
            for pi in range(popSize):
                # update and limit V
                V[pi, :] = 0.4 * V[pi, :] + 2.0 * np.random.random_sample((Dim, )) * (
                    personBestX[pi, :] - X[pi, :]) + 2.0 * np.random.random_sample((Dim, )) * (leaderX - X[pi, :])
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
                if y[pi] < leaderY:
                    leaderY = copy.deepcopy(y[pi])
                    leaderX = copy.deepcopy(X[pi, :])
                    indicatorPoorLeadPower = True

                if y[pi] < personBestY[pi]:
                    personBestX[pi, :] = X[pi, :]
                    personBestY[pi] = y[pi]
                    if y[pi] < gBestY:
                        gBestX = copy.deepcopy(X[pi, :])
                        gBestY = copy.deepcopy(y[pi])
                        indicatorGoodLeadPower = True
                if self.FENum % popSize == 0:
                    self.convergeCurve.append(gBestY)
            if np.sum(personBestY) < np.sum(personBestYBackUp):
                indicatorFairLeadPower = True
            # control the lifespan of the leader
            leaderAge += 1
            if indicatorGoodLeadPower:
                leaderLifeSpan += 2
            elif indicatorFairLeadPower:
                leaderLifeSpan += 1
            elif indicatorPoorLeadPower:
                pass
            else:
                leaderLifeSpan -= 1
            # generate and test the challenger
            if leaderAge >= leaderLifeSpan:
                flagSame = True  # the challenger is the same as the leader
                challenger = copy.deepcopy(leaderX)
                for fd in range(Dim):
                    if np.random.random() < (1 / Dim):
                        challenger[fd] = (
                            upperBoundX[0, fd] - lowerBoundX[0, fd]) * np.random.random_sample() + lowerBoundX[0, fd]
                        flagSame = False
                # ensure that the challenger is different from the leader
                if flagSame:
                    challenger[np.random.randint(Dim)] = (
                        upperBoundX[0, fd] - lowerBoundX[0, fd]) * np.random.random_sample() + lowerBoundX[0, fd]
                XbackUp, VbackUp = copy.deepcopy(X), copy.deepcopy(V)
                start = time.clock()
                challengerY = function(challenger[np.newaxis, :])
                self.FERuntime += (time.clock() - start)
                self.FENum += 1
                if challengerY < gBestY:
                    gBestY = copy.deepcopy(challengerY)
                    gBestX = copy.deepcopy(challenger)
                flagImprove = False
                for ti in range(self.challengingTimesMax):
                    for pi in range(popSize):
                        # update and limit V
                        V[pi, :] = 0.4 * V[pi, :] + 2.0 * np.random.random_sample((Dim, )) * (
                            personBestX[pi, :] - X[pi, :]) + 2.0 * np.random.random_sample((Dim, )) * (challenger - X[pi, :])
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
                        start = time.clock()
                        y[pi] = function(X[pi, :][np.newaxis, :])
                        self.FERuntime += (time.clock() - start)
                        self.FENum += 1
                        if y[pi] < personBestY[pi]:
                            personBestY[pi] = y[pi]
                            personBestX[pi, :] = X[pi, :]
                            flagImprove = True
                            if y[pi] < gBestY:
                                gBestX = copy.deepcopy(X[pi, :])
                                gBestY = copy.deepcopy(y[pi])
                        if y[pi] < challengerY:
                            challengerY = copy.deepcopy(y[pi])
                            challenger = copy.deepcopy(X[pi, :])
                        if self.FENum % popSize == 0:
                            self.convergeCurve.append(gBestY)
                    if flagImprove:
                        leaderX = copy.deepcopy(challenger)
                        leaderY = copy.copy(challengerY)
                        leaderAge = 0
                        leaderLifeSpan = copy.copy(self.leaderLifeSpanMax)
                        break
                if not flagImprove:
                    X = copy.deepcopy(XbackUp)
                    V = copy.deepcopy(VbackUp)
                    leaderAge = leaderLifeSpan - 1
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
