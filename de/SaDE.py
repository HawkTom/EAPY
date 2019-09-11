# !/usr/bin/env python
"""
SaDE.py

Description:       the implemention of modified simple SaDE

Refrence paper:
Qin, A. Kai, and Ponnuthurai N. Suganthan.
"Self-adaptive differential evolution algorithm for numerical optimization."

Member variables:

     Name:                      SaDE
     FERuntime:                 time for fitness evaluation
     FENum:                     number of fitness evaluation
     runtime:                   time for whole algorithm
     optimalX:                  optimal solution for problem
     optimalY:                  optimal value for problem
     convergeCurve:             the procedure of convergence
     convergeCurveInterval:     inverval between two saved points
     CRm:                       crossover constant
     p1:                        the probability of applying strategy one  


Member function:
     
     setParameters(weight, learningRate):   setting parameters

     optimize(cfp, ap, printLog):           the main process of optimization

                                            cfp:        config for continue function parameters
                                            ap:         config for algorithm parameters
                                            printLog:   determine whether to print Log after opitmization
                                                        (true default)

Example: 

    agent = SaDE()
    agent.optimize(cfp, ap, printLog=True) # cfp ap need to config at first

"""

from function import continueFunction as cF
import numpy as np
import time
import sys
import copy

__all__ = ['SaDE']


class SaDE:

    def __init__(self):
        self.Name = "SaDE"
        self.FERuntime = 0
        self.FENum = 0
        self.setParameters()

    # setting F, CR
    def setParameters(self, p1=0.5, CRm=0.5):
        self.p1 = p1
        self.CRm = CRm

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
        gen = 0
        ns1, nf1 = 0, 0
        ns2, nf2 = 0, 0
        setCR = []
        while self.FENum < ap.FEMax:
            Xnew = copy.deepcopy(X)
            gBestX = copy.deepcopy(X[0, :])
            # update the CR value            
            if gen % 5 == 0:
                CR = np.zeros((popSize, ))
                for i in range(popSize):
                    tempCR = np.random.normal(self.CRm, 0.1)
                    while tempCR > 1 or tempCR < 0:
                        tempCR = np.random.normal(self.CRm, 0.1)
                    CR[i] = tempCR
            for pi in range(popSize):
                ind = X[pi, :]
                trail = np.zeros_like(ind)
                F = np.random.normal(0.5, 0.3)
                # limit the F in (0.5, 2]
                while F > 2 or F <= 0:
                    F = np.random.normal(0.5, 0.3)
                # random select 3 vectors different pi
                r = np.random.randint(popSize, size=(3, ))
                while (r == pi).any():
                    r = np.random.randint(popSize, size=(3, ))
                # mutation
                if np.random.random_sample() < self.p1:
                    v = X[r[0], :] + F * (X[r[1], :] - X[r[2], :])
                    selectFlag = 1
                else:
                    v = ind + F * (gBestX - ind) + F * (X[r[0], :] - X[r[1], :])
                    selectFlag = 2
                # crossover
                jPosition = np.random.randint(Dim)
                Position = np.random.random_sample(size=(Dim, )) < CR[pi]
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
                    setCR.append(CR[pi])
                    if selectFlag == 1:
                        ns1 += 1
                    else:
                        ns2 += 1
                else:
                    if selectFlag == 1:
                        nf1 += 1
                    else:
                        nf2 += 1
            if gen % 25 == 0 and len(setCR) != 0:
                self.CRm = np.mean(setCR)
                setCR = []
            if gen % 50 == 0:
                if ( ns2*(ns1+nf1) + ns1*(ns2+nf2) ) == 0:
                    self.p1 = 0.01
                else:                    
                    self.p1 = ns1*(ns2+nf2) / ( ns2*(ns1+nf1) + ns1*(ns2+nf2) ) + 0.01
                ns1, nf1 = 0, 0
                ns2, nf2 = 0, 0
            indexSort = np.argsort(y)
            y.sort()
            Xnew = Xnew[indexSort, :]
            self.convergeCurve.append(y[0])
            X = copy.deepcopy(Xnew)
            gen += 1
        self.optimalX = X[0, :]
        self.optimalY = y[0]
        self.convergeCurveIntrval = popSize
        if printLog:
            # summary
            print('$--------Result--------$\n')
            print('*Function: {0}\tDimension: {1}\t FEMax: {2}\n'.format(
                cfp.funcName, cfp.funcDim, self.FENum))
            print('Optimal Y  : {0} \n'.format(self.optimalY))        