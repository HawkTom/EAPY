#!/usr/bin/env python
"""
implement of PSO algorithm: CLPSO, wPSO, ALCPSO
"""

__authors__ = "Hao Tong"
__date__ = "2018/03/28"
__version__ = "0.0.1"

from Function import continueFunction as cF
import numpy as np
import time
import sys
import copy


class CLPSO:

    def __init__(self, cfp, ap):
        self.Name = "CLPSO"
        self.FERuntime = 0
        self.FENum = 0
        runtimeStart = time.clock()
        self.main(cfp, ap)
        self.runtime = time.clock() - runtimeStart

    def main(self, cfp, ap):
        np.random.seed(ap.initialSeed)
        popSize = ap.populationSize
        Dim = cfp.functionDimension
        function = getattr(cF, cfp.functionName)
        lowBoundX = np.kron(np.ones((popSize, 1)), cfp.functionBound[0, :])
        upperBoundX = np.kron(np.ones((popSize, 1)), cfp.functionBound[1, :])
        upperBoundV = 0.2 * (upperBoundX - lowBoundX)
        lowBoundV = -1 * upperBoundV
        X = (upperBoundX - lowBoundX) * \
             np.random.random_sample((popSize, Dim)) + lowBoundX
        V = (upperBoundV - lowBoundV) * \
             np.random.random_sample((popSize, Dim)) + lowBoundV
        start = time.clock()
        y = function(X)
        self.FERuntime += (time.clock()-start)
        self.FENum += popSize
        # initialize personal best X and y
        personBestX, personBestY = copy.deepcopy(X), copy.deepcopy(y)
        # initialize global best X and y
        gBestX, gBestY = X[np.argmin(y), :], np.min(y)
        self.convergeCurve = [y[0], gBestY]
        # initialize learn probability and exemplar learned for each particle
        learnX = personBestX
        # initialize weight, refresh gap, learning rate
        w = np.array([0.9, 0.4])
        refreshingGap = 7
        learnProbability = 0.05 + 0.45 * \
            (np.exp(10 * (np.array(range(1, popSize+1)) - 1) /
             (popSize - 1)) - 1) / (np.exp(10) - 1)
        learningRate = 1.4995
        refreshGaps = np.zeros((popSize, 1))
        maxGen, gen = ap.iterationMax, 0
        while self.FENum < ap.FEMax:
            wk = w[0] - (w[0] - w[1]) * gen / maxGen
            for pi in range(popSize):
                # allow the particle to learn from the exemplar until the particle
                # stops improving for a certain number of generations
                if refreshGaps[pi] >= refreshingGap:
                    refreshGaps[pi] = 0
                    learnFlag = False  # flag to learn to at least other particle
                    for fd in range(Dim):
                        if np.random.random() < learnProbability[pi]:
                            aOrb = np.random.permutation(popSize)
                            if personBestY[aOrb[1]] < personBestY[aOrb[0]]:
                                learnX[pi, fd] = personBestX[aOrb[1], fd]
                            else:
                                learnX[pi, fd] = personBestX[aOrb[0], fd]
                            learningFlag = True
                        else:
                            learnX[pi, fd] = personBestX[pi, fd]
                    # make sure to learn to at least other particle for one random dimension
                    if not learnFlag:
                        fd = np.random.randint(Dim)
                        aOrb = np.random.permutation(popSize)
                        if aOrb[0] == pi:
                            exemplar = aOrb[1]
                        else:
                            exemplar = aOrb[0]
                        learnX[pi, fd] = personBestX[exemplar, fd]
                # update and limit V
                V[pi, :] = wk * V[pi, :] + learningRate * \
                    np.random.random_sample(
                        (1, Dim)) * (learnX[pi, :] - X[pi, :])
                V[pi, :][V[pi, :] < lowBoundV[pi, :]
                    ] = lowBoundV[pi, :][V[pi, :] < lowBoundV[pi, :]]
                V[pi, :][V[pi, :] > upperBoundV[pi, :]
                    ] = upperBoundV[pi, :][V[pi, :] > upperBoundV[pi, :]]
                # update X
                X[pi, :] = X[pi, :] + V[pi, :]
                # update personal and global best X and y
                if (X[pi, :] > lowBoundX[pi, :]).all() & (X[pi, :] < upperBoundX[pi, :]).all():
                    start = time.clock()
                    # print(X[pi, :])
                    y[pi] = function(X[pi, :][np.newaxis, :])
                    self.FERuntime += (time.clock() - start)
                    self.FENum += 1
                    if y[pi] < personBestY[pi]:
                        personBestX[pi, :] = X[pi, :]
                        personBestY[pi] = y[pi]
                        refreshGaps[pi] = 0
                        if personBestY[pi] < gBestY:
                            gBestX = personBestX[pi, :]
                            gBestY = personBestY[pi]
                    else:
                        refreshGaps[pi] += 1
                    if self.FENum % popSize == 0:
                        self.convergeCurve.append(gBestY)
            gen = gen + 1
            # print(X)
            # print('Gen:{0} BestV: {1} \n'.format(self.FENum, gBestY))
        self.optimalX = gBestX
        self.optimalY = gBestY
        self.convergeCurveIntrval = popSize


class wPSO:

    def __init__(self, cfp, ap):
        self.Name = "wPSO"
        self.FERuntime = 0
        self.FENum = 0
        runtimeStart = time.clock()
        self.main(cfp, ap)
        self.runtime = time.clock() - runtimeStart

    def main(self, cfp, ap):
        np.random.seed(ap.initialSeed)
        popSize = ap.populationSize
        Dim = cfp.functionDimension
        function = getattr(cF, cfp.functionName)
        lowBoundX = np.kron(np.ones((popSize, 1)), cfp.functionBound[0, :])
        upperBoundX = np.kron(np.ones((popSize, 1)), cfp.functionBound[1, :])
        upperBoundV = 0.2 * (upperBoundX - lowBoundX)
        lowBoundV = -1 * upperBoundV
        X = (upperBoundX - lowBoundX) * \
             np.random.random_sample((popSize, Dim)) + lowBoundX
        V = (upperBoundV - lowBoundV) * \
             np.random.random_sample((popSize, Dim)) + lowBoundV
        start = time.clock()
        y = function(X)
        self.FERuntime += (time.clock()-start)
        self.FENum += popSize
        # initialize personal best X and y
        personBestX, personBestY = copy.deepcopy(X), copy.deepcopy(y)
        # initialize global best X and y
        gBestX, gBestY = X[np.argmin(y), :], np.min(y)
        self.convergeCurve = [y[0], gBestY]
        # initialize weight, refresh gap, learning rate
        w = np.array([0.9, 0.4])
        learningRate = 2.0
        maxGen, gen = ap.iterationMax, 0
        while self.FENum < ap.FEMax:
            wk = w[0] - (w[0] - w[1]) * gen / maxGen
            for pi in range(popSize):
                # update and limit V
                V[pi, :] = wk * V[pi, :] + learningRate * np.random.random_sample((1, Dim)) * (personBestX[pi, :] - X[pi, :]) + learningRate * np.random.random_sample((1, Dim)) * (gBestX - X[pi, :])
                V[pi, :][V[pi, :] < lowBoundV[pi, :]
                    ] = lowBoundV[pi, :][V[pi, :] < lowBoundV[pi, :]]
                V[pi, :][V[pi, :] > upperBoundV[pi, :]
                    ] = upperBoundV[pi, :][V[pi, :] > upperBoundV[pi, :]]
                # update X
                X[pi, : ] = X[pi, : ] + V[pi, : ]
                X[pi, :][X[pi, :] < lowBoundX[pi, :]] = lowBoundX[pi, :][X[pi, :] < lowBoundX[pi, :]]
                X[pi, :][X[pi, :] > upperBoundX[pi, :]] = upperBoundX[pi, :][X[pi, :] > upperBoundX[pi, :]]
                # update personal and global best X and y
                start= time.clock()
                # print(X[pi, :])
                y[pi] = function(X[pi, :][np.newaxis, : ])
                self.FERuntime += (time.clock() - start)
                self.FENum += 1
                if y[pi] < personBestY[pi]:
                    personBestX[pi, :] = X[pi, : ]
                    personBestY[pi]= y[pi]
                    if personBestY[pi] < gBestY:
                        gBestX = personBestX[pi, : ]
                        gBestY= personBestY[pi]
                if self.FENum % popSize == 0:
                    self.convergeCurve.append(gBestY)
            gen= gen + 1
            # print(X)
            print('Gen:{0} BestV: {1} \n'.format(self.FENum, gBestY))
        self.optimalX= gBestX
        self.optimalY= gBestY
        self.convergeCurveIntrval= popSize


class ALCPSO:

    def __init__(self, cfp, ap):
        self.Name = "ALCPSO"
        self.FERuntime = 0
        self.FENum = 0
        runtimeStart = time.clock()
        self.main(cfp, ap)
        self.runtime = time.clock() - runtimeStart

    def main(self, cfp, ap):
        np.random.seed(ap.initialSeed)
        popSize = ap.populationSize
        Dim = cfp.functionDimension
        function = getattr(cF, cfp.functionName)
        lowBoundX = np.kron(np.ones((popSize, 1)), cfp.functionBound[0, :])
        upperBoundX = np.kron(np.ones((popSize, 1)), cfp.functionBound[1, :])
        upperBoundV = 0.2 * (upperBoundX - lowBoundX)
        lowBoundV = -1 * upperBoundV
        X = (upperBoundX - lowBoundX) * \
            np.random.random_sample((popSize, Dim)) + lowBoundX
        V = (upperBoundV - lowBoundV) * \
            np.random.random_sample((popSize, Dim)) + lowBoundV
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
        challengingTimesMax = 2
        leaderLifeSpanMax = 60
        leaderLifeSpan = copy.copy(leaderLifeSpanMax)
        while self.FENum < ap.FEMax:
            indicatorGoodLeadPower = False
            indicatorFairLeadPower = False
            indicatorPoorLeadPower = False
            personBestYBackUp = copy.deepcopy(personBestY)
            for pi in range(popSize):
                # update and limit V
                V[pi, :] = 0.4 * V[pi, :] + 2.0 * np.random.random_sample((1, Dim)) * (
                    personBestX[pi, :] - X[pi, :]) + 2.0 * np.random.random_sample((1, Dim)) * (leaderX - X[pi, :])
                V[pi, :][V[pi, :] < lowBoundV[pi, :]
                         ] = lowBoundV[pi, :][V[pi, :] < lowBoundV[pi, :]]
                V[pi, :][V[pi, :] > upperBoundV[pi, :]
                         ] = upperBoundV[pi, :][V[pi, :] > upperBoundV[pi, :]]
                # update X
                X[pi, :] = X[pi, :] + V[pi, :]
                # update personal and global best X and y
                start= time.clock()
                y[pi] = function(X[pi, :][np.newaxis, : ])
                self.FERuntime += (time.clock() - start)
                self.FENum += 1
                if y[pi] < leaderY:
                    leaderY = y[pi]
                    leaderX = X[pi, :]
                    indicatorPoorLeadPower = True

                if y[pi] < personBestY[pi]:
                    personBestX[pi, :] = X[pi, : ]
                    personBestY[pi]= y[pi]
                    if personBestY[pi] < gBestY:
                        gBestX = personBestX[pi, : ]
                        gBestY= personBestY[pi]
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
                flagSame = True # the challenger is the same as the leader
                challenger = copy.deepcopy(leaderX)
                for fd in range(Dim):
                    if np.random.random < (1 / Dim):
                        challenger[fd] = (upperBoundX[1, fd] - lowBoundX[1, fd]) * np.random.random_sample() + lowBoundX[1, fd]
                        flagSame = False
                # ensure that the challenger is different from the leader
                if flagSame:
                    challenger[np.random.randint[Dim]] = (upperBoundX[1, fd] - lowBoundX[1, fd]) * np.random.random_sample() + lowBoundX[1, fd]
                XbackUp, VbackUp = copy.deepcopy(X), copy.deepcopy(V)
                start= time.clock()
                challengerY = function(challenger[np.newaxis, : ])
                self.FERuntime += (time.clock() - start)
                self.FENum += 1
                if challengerY < gBestY:
                    gBestY = copy.deepcopy(challengerY)
                    gBestX = copy.deepcopy(challenger)
                flagImprove = False
                for ti in range(challengingTimesMax):
                    for pi in range(popSize):
                             # update and limit V
                        V[pi, :] = 0.4 * V[pi, :] + 2.0 * np.random.random_sample((1, Dim)) * (
                            personBestX[pi, :] - X[pi, :]) + 2.0 * np.random.random_sample((1, Dim)) * (challenger - X[pi, :])
                        V[pi, :][V[pi, :] < lowBoundV[pi, :]
                                ] = lowBoundV[pi, :][V[pi, :] < lowBoundV[pi, :]]
                        V[pi, :][V[pi, :] > upperBoundV[pi, :]
                                ] = upperBoundV[pi, :][V[pi, :] > upperBoundV[pi, :]]
                        # update X
                        X[pi, :] = X[pi, :] + V[pi, :]
                        start= time.clock()
                        y[pi] = function(X[pi, :][np.newaxis, : ])
                        self.FERuntime += (time.clock() - start)
                        self.FENum += 1
                        if y[pi] < personBestY[pi]:
                            personBestY = y[pi]
                            personBestX = X[pi, :]
                            flagImprove = True
                            if y[pi] < gBestY:
                                gBestX = X[pi, :]
                                gBestY = y[pi]
                        if y[pi] < gBestY:
                            challengerY = y[pi]
                            challenger = X[pi, :]
                        if self.FENum % popSize == 0:
                            self.convergeCurve.append(gBestY)
                    if flagImprove:
                        leaderX = challenger
                        leaderY = challengerY
                        leaderAge = 0
                        leaderLifeSpan = copy.copy(leaderLifeSpanMax)
                        break
                if not flagImprove:
                    X = XbackUp
                    V = VbackUp
                    leaderAge -= 1
                    
            # print('Gen:{0} BestV: {1} \n'.format(self.FENum, gBestY))
        self.optimalX = gBestX
        self.optimalY = gBestY
        self.convergeCurveIntrval = popSize
