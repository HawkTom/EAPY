#!/usr/bin/env python
"""
implement of sDE algorithm
"""

__authors__ = "Hao Tong"
__date__ = "2018/04/02"
__version__ = "0.0.1"

from Function import continueFunction as cF
import numpy as np
import time
import sys
import copy

class sDE:

    def __init__(self, cfp, ap):
        self.Name = "sDE"
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
        X = (upperBoundX - lowBoundX) * \
            np.random.random_sample((popSize, Dim)) + lowBoundX
        start = time.clock()
        y = function(X)
        self.FERuntime += (time.clock()-start)
        self.FENum += popSize
        self.convergeCurve = [y[0], np.min(y)]
        # parameters
        CR, F = 0.1, 0.5
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
                v = X[r[0], :] + F * (X[r[1], :] - X[r[2], :])
                # crossover
                jPosition = np.random.randint(Dim)
                Position = np.random.random_sample(size=(Dim, )) < CR
                Position[jPosition] = True
                trail[Position] = v[Position]
                trail[~Position] = ind[~Position]
                trail[trail < lowBoundX[pi, :]] = lowBoundX[pi, :][trail < lowBoundX[pi, :]]
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
            # print('Gen:{0} BestV: {1} \n'.format(self.FENum, gBestY))
        self.optimalX = gBestX
        self.optimalY = gBestY
        self.convergeCurveIntrval = popSize


class JADE:

    def __init__(self, cfp, ap):
        self.Name = "JADE"
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
        X = (upperBoundX - lowBoundX) * \
            np.random.random_sample((popSize, Dim)) + lowBoundX
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
        muF, muCR = 0.5, 0.5
        p, c = 0.05, 0.1
        A = [] # denote A as the set of archived inferior solutions
        # main loop
        while self.FENum < ap.FEMax:
            Xnew = copy.deepcopy(X)
            setF, setCR = [], []
            for pi in range(popSize):
                CR = np.random.normal(muCR, 0.1)
                F = min(muF + 0.1 * np.tan(np.pi * (np.random.random_sample() - 0.5)), 1)
                ind = X[pi, :]
                trail = np.zeros_like(ind)
                # mutation
                xBestP = X[np.random.randint(popSize * p), :]
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
                trail[trail < lowBoundX[pi, :]] = lowBoundX[pi, :][trail < lowBoundX[pi, :]]
                trail[trail > upperBoundX[pi, :]] = upperBoundX[pi, :][trail > upperBoundX[pi, :]]
                start = time.clock()
                fitnessTrail = function(trail[np.newaxis, : ])
                self.FERuntime += (time.clock() - start)
                self.FENum += 1
                if y[pi] > fitnessTrail[0]:
                    Xnew[pi, :] = trail
                    y[pi] = fitnessTrail[0]
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
                muCR = (1 - c) * muCR + c * np.mean(setCR)
                muF = (1 - c) * muF + c * (np.sum(np.array(setF)**2) / np.sum(setF))
            indexSort = np.argsort(y)
            y.sort()
            Xnew = Xnew[indexSort, :]
            self.convergeCurve.append(y[0])
            X = copy.deepcopy(Xnew)
        self.optimalX = X[0, :]
        self.optimalY = y[0]
        self.convergeCurveIntrval = popSize


class SaDE:

    def __init__(self, cfp, ap):
        self.Name = "SaDE"
        self.FERuntime = 0
        self.FENum = 0
        runtimeStart = time.clock()
        self.main(cfp, ap)
        self.runtime = time.clock() - runtimeStart

    def main(self, cfp, ap):
        # np.random.seed(ap.initialSeed)
        popSize = ap.populationSize
        Dim = cfp.functionDimension
        function = getattr(cF, cfp.functionName)
        lowBoundX = np.kron(np.ones((popSize, 1)), cfp.functionBound[0, :])
        upperBoundX = np.kron(np.ones((popSize, 1)), cfp.functionBound[1, :])
        X = (upperBoundX - lowBoundX) * \
            np.random.random_sample((popSize, Dim)) + lowBoundX
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
        CRm, p1 = 0.5, 0.5
        gen = 0
        ns1, nf1 = 0, 0
        ns2, nf2 = 0, 0
        setCR = []
        while self.FENum < ap.FEMax:
            # if gen == 775:
            #     print('Error')
            Xnew = copy.deepcopy(X)
            gBestX = X[0, :]
            # update the CR value
            CR = np.zeros((popSize, ))
            if gen % 5 == 0:
                for i in range(popSize):
                    tempCR = np.random.normal(CRm, 0.1)
                    while tempCR > 1 or tempCR < 0:
                        tempCR = np.random.normal(CRm, 0.1)
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
                if np.random.random_sample() < p1:
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
                trail[trail < lowBoundX[pi, :]] = lowBoundX[pi, :][trail < lowBoundX[pi, :]]
                trail[trail > upperBoundX[pi, :]] = upperBoundX[pi, :][trail > upperBoundX[pi, :]]
                start = time.clock()
                fitnessTrail = function(trail[np.newaxis, : ])
                self.FERuntime += (time.clock() - start)
                self.FENum += 1
                if fitnessTrail[0] < y[pi]:
                    Xnew[pi, :] = trail
                    y[pi] = fitnessTrail[0]
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
                CRm = np.mean(setCR)
                setCR = []
            if gen > 50:
                if ( ns2*(ns1+nf1) + ns1*(ns2+nf2) ) == 0:
                    p1 = 0.01
                else:
                    # if ( ns2*(ns1+nf1) + ns1*(ns2+nf2) ) == 0:
                    #     print("division by 0")
                    
                    p1 = ns1*(ns2+nf2) / ( ns2*(ns1+nf1) + ns1*(ns2+nf2) ) + 0.01
                ns1, nf1 = 0, 0
                ns2, nf2 = 0, 0
            indexSort = np.argsort(y)
            np.sort(y)
            Xnew = Xnew[indexSort, :]
            self.convergeCurve.append(y[0])
            X = copy.deepcopy(Xnew)
            gen += 1
            # print(y[0])
        self.optimalX = X[0, :]
        self.optimalY = y[0]
        self.convergeCurveIntrval = popSize
        

        

