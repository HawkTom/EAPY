#!/usr/bin/env python
"""
suitrun.py

Description:  run a single algorithm for several times 

Class: executer

member variables: 
        
        function parameters
        trial parameters
        algorithms

function: give the statistic information of several runing

"""
from pso import *
from de import *
import numpy as np
import sys, os
from statistic import printSTAT
import time

class executer:

    def __init__(self, cfp, tp, ap):
        # CFP: functionName, functionBound, functionDimension, functionInitialBound, target, opitmalX, opitmalY
        self.cfp = cfp
        # TP: testNumber, testLog, testSeed, testSeeds
        self.tp = tp
        # AP: algorithmName, iterationMax, initialSeed, populationSize, FEMax, other algorithm parameters
        self.ap = ap

    def run(self, filePath = None):
        
        algorithm = getattr(sys.modules[__name__], self.ap.algorithmName)
        optimalRes = []
        record = {}
        record['optimalYs'], record['FuncEvalRuntimes'] = [], []
        record['Runtimes'], record['FuncEvalNums'] = [], []    
        record['optimalXs'] = []    

        runInfo = '* Function: {0}\tRuns: {1}\tDimension: {2}\n'.format(
            self.cfp.funcName, self.tp.testNumber, self.cfp.funcDim)
        algorithmInfo = '* Algorithm: {0}\tFEMax: {1}\n'.format(self.ap.algorithmName, self.ap.FEMax)
        print(runInfo)
        print(algorithmInfo)

        for iteration in range(self.tp.testNumber): 
            self.ap.initialSeed = self.tp.testSeeds[iteration]
            # run algorithm for one iteration
            agent = algorithm()
            agent.optimize(self.cfp, self.ap, printLog=False)
            optimalRes.append(agent)
            record['optimalYs'].append(agent.optimalY)
            record['Runtimes'].append(agent.runtime)
            record['FuncEvalRuntimes'].append(agent.FERuntime)
            record['FuncEvalNums'].append(agent.FENum)
            record['optimalXs'].append('run' + str(iteration) + '\t' + '\t'.join([str(_) for _ in agent.optimalX]) + '\n')
            if self.tp.testLog:
                print('run {0}: y = {1} || runtime = {2} || FuncEvalRuntime = {3} || FuncEvalNumber = {4} \n'
                      .format(iteration, agent.optimalY, agent.runtime, agent.FERuntime, agent.FENum))
        if filePath:
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            fileName = filePath + self.ap.algorithmName + '_' + self.cfp.funcName + '_' + str(self.tp.testSeed) + '.dat'
            with open(fileName, 'w+') as f:
                f.write(runInfo)
                f.write(algorithmInfo)
                runIndex = '\n\t'
                for iteration in range(self.tp.testNumber):
                    runIndex = runIndex + 'run' + str(iteration) + '\t'
                f.write(runIndex + '\n')
                for item in record:
                    f.write(item + ':\t' + '\t'.join([str(_) for _ in record[item]]) + '\n')

        # summary
        print('$--------Running Information--------$\n')
        print('* Function: {0}\tRuns: {1}\tDimension: {2}\n'.format(
            self.cfp.funcName, self.tp.testNumber, self.cfp.funcDim))
        print(
            '* Algorithm: {0}\tFEMax: {1}\n'.format(agent.Name, self.ap.FEMax))
        printSTAT(record)
