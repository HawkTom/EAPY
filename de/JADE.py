# !/usr/bin/env python
"""
JADE.py

Description:       the implemention of modified JADE

Refrence paper:
Zhang, Jingqiao, and Arthur C. Sanderson. 
"JADE: adaptive differential evolution with optional external archive." 

Member variables:

     Name:                      JADE
     FERuntime:                 time for fitness evaluation
     FENum:                     number of fitness evaluation
     runtime:                   time for whole algorithm
     optimalX:                  optimal solution for problem
     optimalY:                  optimal value for problem
     convergeCurve:             the procedure of convergence
     convergeCurveInterval:     inverval between two saved points
     muF:                       the initial mean of F distribution
     muCR:                      the initial mean of CR distribution
     p:                         top p inidividuals
     c:                         postive constant for muCR



Member function:
     
     setParameters(weight, learningRate):   setting parameters

     optimize(cfp, ap, printLog):           the main process of optimization

                                            cfp:        config for continue function parameters
                                            ap:         config for algorithm parameters
                                            printLog:   determine whether to print Log after opitmization
                                                        (true default)

Example: 

    agent = JADE()
    agent.optimize(cfp, ap, printLog=True) # cfp ap need to config at first

"""

from function import continueFunction as cF
import numpy as np
import time
import sys
import copy

__all__ = ['JADE']


class JADE:

    def __init__(self):
        self.Name = "JADE"
        self.FERuntime = 0
        self.FENum = 0
        self.setParameters()

    # setting muF, muCR, p, c
    def setParameters(self, muF=0.5, muCR=0.5, p=0.05, c=0.1):
        self.muF = muF
        self.muCR = muCR
        self.p = p
        self.c = c

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

        # initial X position
        X = (upperInitBoundX - lowerInitBoundX) * \
            np.random.random_sample((popSize, Dim)) + lowerInitBoundX
        start = time.clock()
        y = function(X)
        self.FERuntime += (time.clock()-start)
        self.FENum += popSize
        self.convergeCurve = [y[0]]
        indexSort = np.argsort(y)
        y.sort()
        X = X[indexSort, :]
        self.convergeCurve.append(y[0])
        # parameters
        A = [] # denote A as the set of archived inferior solutions
        # main loop
        gen = 0
        while self.FENum < ap.FEMax:
            gen += 1
            Xnew = copy.deepcopy(X)
            setF, setCR = [], []
            for pi in range(popSize):
                CR = min(max(np.random.normal(self.muCR, 0.1), 0), 1)
                F = min(self.muF + 0.1 * np.tan(np.pi * (np.random.random_sample() - 0.5)), 1)
                while F <= 0:
                    F = min(self.muF + 0.1 * np.tan(np.pi * (np.random.random_sample() - 0.5)), 1)
                ind = X[pi, :]
                trail = np.zeros_like(ind)
                # mutation
                xBestP = X[np.random.randint(popSize * self.p), :]
                r1 = np.random.randint(popSize)
                while r1 == pi:
                    r1 = np.random.randint(popSize)
                r1X = X[r1, :]
                if len(A) != 0:
                    U = np.vstack((X, np.array(A)))
                else:
                    U = X
                r2 = np.random.randint(popSize)
                while r2 == pi or r2 == r1:
                   r2 = np.random.randint(U.shape[0])
                r2X = U[r2, :]
                v = ind + F * (xBestP - ind) + F * (r1X - r2X)
                # crossover
                jPosition = np.random.randint(Dim)
                Position = np.random.random_sample(size=(Dim, )) < CR
                Position[jPosition] = True
                trail[Position] = v[Position]
                trail[~Position] = ind[~Position]
                trail[trail < lowerBoundX[pi, :]] = lowerBoundX[pi, :][trail < lowerBoundX[pi, :]]
                trail[trail > upperBoundX[pi, :]] = upperBoundX[pi, :][trail > upperBoundX[pi, :]]
                start = time.clock()
                fitnessTrail = function(trail[np.newaxis, : ])
                self.FERuntime += (time.clock() - start)
                self.FENum += 1
                if y[pi] > fitnessTrail:
                    Xnew[pi, :] = trail
                    y[pi] = fitnessTrail
                    A.append(ind)
                    setCR.append(CR)
                    setF.append(F)
            # randomly remove solution from A so that |A| <= NP
            sizeA = len(A)
            while sizeA > popSize:
                del A[np.random.randint(sizeA)]
                sizeA = len(A)
            #update CR and F
            if len(setCR) != 0 and np.sum(setF) > 0:
                self.muCR = (1 - self.c) * self.muCR + self.c * np.mean(setCR)
                self.muF = (1 - self.c) * self.muF + self.c * (np.sum(np.array(setF)**2) / np.sum(setF))
            indexSort = np.argsort(y)
            y.sort()
            Xnew = Xnew[indexSort, :]
            self.convergeCurve.append(y[0])
            X = copy.deepcopy(Xnew)
            
        self.optimalX = X[0, :]
        self.optimalY = y[0]
        self.convergeCurveIntrval = popSize
        if printLog:
            # summary
            print('$--------Result--------$\n')
            print('*Function: {0}\tDimension: {1}\t FEMax: {2}\n'.format(
                cfp.funcName, cfp.funcDim, self.FENum))
            print('Optimal Y  : {0} \n'.format(self.optimalY))