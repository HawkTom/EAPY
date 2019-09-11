#!/usr/bin/env python
"""

"""

import config
from suiterun import executer
import time
from de import sDE, JADE, SaDE
from ga.GL25 import GL25
from es.cmaes import cmaes




class Example:

    def __init__(self):
        # CFP: funcName, funcDim, funcLowerBound, funcUpperBound, 
        #      funcInitLowerBound, funcInitUpperBound, target, opitmalX, opitmalY
        cfp = config.ContinueFunctionParameter('sphere', 30, -100, 100)
        # TP: testNumber, testLog, testSeed, testSeeds
        tp = config.TrialParameter(testNumber=1)
        # AP: algorithmName, iterationMax, initialSeed, populationSize, FEMax, other algorithm parameters
        ap = config.AlgorithmParameter(algorithmName='sDE',
                                FuncEvalMax=10000, populationSize=100)
        # run algorihtm in several runs                                
        # optimizer = executer(cfp, tp, ap)
        # optimizer.run()
        # run algorihtm on time
        agent = sDE()
        agent.optimize(cfp, ap, printLog=True)


def main():
    start = time.clock()
    a = Example()
    period = time.clock() - start
    print('Total running time: {0} seconds'.format(period))

if __name__ == '__main__':
    main()
