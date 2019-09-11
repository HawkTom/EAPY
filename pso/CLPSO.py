# !/usr/bin/env python
"""
CLPSO.py

Description:      the implemention of CLPSO

Refrence paper:
Liang, Jing J., et al. "Comprehensive learning particle swarm optimizer for global optimization of multimodal functions." 
IEEE transactions on evolutionary computation 10.3 (2006): 281-295.

Member variables:

     Name:                      CLPSO
     FERuntime:                 time for fitness evaluation
     FENum:                     number of fitness evaluation
     runtime:                   time for whole algorithm
     optimalX:                  optimal solution for problem
     optimalY:                  optimal value for problem
     convergeCurve:             the procedure of convergence
     convergeCurveInterval:     inverval between two saved points
     w(weight):                 default w_max = 0.9, w_min = 0.4 
     learningRate:              default = 1.4995
     refreshingGap:             default = 7

Member function:
     
     setParameters(weight, learningRate):   setting parameters

     optimize(cfp, ap, printLog):           the main process of optimization

                                            cfp:        config for continue function parameters
                                            ap:         config for algorithm parameters
                                            printLog:   determine whether to print Log after opitmization
                                                        (true default)

Example: 

    agent = CLPSO()
    agent.optimize(cfp, ap, printLog=True) # cfp ap need to config at first

"""

from function import continueFunction as cF
import numpy as np
import time
import sys
import copy


class CLPSO:

    def __init__(self):
        self.Name = "CLPSO"
        self.FERuntime = 0
        self.FENum = 0
        self.setParameters()

    def setParameters(self, weight=[0.9, 0.4], learningRate=1.4995,refreshingGap = 7):
        self.w = weight
        self.learningRate = learningRate
        self.refreshingGap = refreshingGap

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

        lowerInitBoundX = np.kron(np.ones((popSize, 1)), cfp.funcInitLowerBound)
        upperInitBoundX = np.kron(np.ones((popSize, 1)), cfp.funcInitUpperBound)

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
        # initialize learn probability and exemplar learned for each particle
        learnX = copy.deepcopy(personBestX)
        # initialize weight, refresh gap, learning rate
        learnProbability = 0.05 + 0.45 * \
            (np.exp(10 * (np.array(range(1, popSize+1)) - 1) /
                    (popSize - 1)) - 1) / (np.exp(10) - 1)
        refreshGaps = np.zeros((popSize, 1))
        maxGen, gen = ap.iterationMax, 0
        while self.FENum < ap.FEMax:
            wk = self.w[0] - (self.w[0] - self.w[1]) * gen / maxGen
            for pi in range(popSize):
                # allow the particle to learn from the exemplar until the particle
                # stops improving for a certain number of generations
                if refreshGaps[pi] >= self.refreshingGap:
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
                V[pi, :] = wk * V[pi, :] + self.learningRate * \
                    np.random.random_sample(
                        (1, Dim)) * (learnX[pi, :] - X[pi, :])
                V[pi, :][V[pi, :] < lowerBoundV[pi, :]
                         ] = lowerBoundV[pi, :][V[pi, :] < lowerBoundV[pi, :]]
                V[pi, :][V[pi, :] > upperBoundV[pi, :]
                         ] = upperBoundV[pi, :][V[pi, :] > upperBoundV[pi, :]]
                # update X
                X[pi, :] = X[pi, :] + V[pi, :]
                # update personal and global best X and y
                if (X[pi, :] > lowerBoundX[pi, :]).all() & (X[pi, :] < upperBoundX[pi, :]).all():
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
            # print('Gen:{0} BestV: {1} \n'.format(self.FENum, gBestY))
        self.optimalX = gBestX
        self.optimalY = gBestY
        self.convergeCurveIntrval = popSize
        if printLog:
            # summary
            print('$--------Result--------$\n')
            print('*Function: {0}\tDimension: {1}\t FEMax: {2}\n'.format(cfp.funcName, cfp.funcDim, self.FENum))
            print('Optimal Y  : {0} \n'.format(self.optimalY))
