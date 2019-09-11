#!/usr/bin/env python
"""
statistic.py

Description:    statistic information for several runs 


printSTAT:  

        input is a dictionary for results in several tests: 
          result['FunEvalRuntimes']:    a list for function evaluation runtime in each test
          result['Runtimes']:           a list for total runtime in each test
          result['FunEvalNums']:        a list for function evaluation numbers in each test
          result['optimalYs']:          a list for optimal values obtained in each test

        FuncEvalRatio: the proportion of function evaluation in whole runtime


"""

import numpy as np

__all__ = ['printSTAT']

def printSTAT(result):
    # result: dictionary 
    # result[FunEvalRuntimes]
    # result[Runtimes]
    # result[FunEvalNums]
    # result[optimals]
    #        result['optimalYs'], result['FuncEvalRuntimes'] = [], []
    #       result['Runtimes'], result['FuncEvalNums'] = [], []
    print('$--------Statistic Information--------$\n')
    FuncEvalRatio = 100.0 * np.array(result['FuncEvalRuntimes']) / np.array(result['Runtimes'])
    print('Optimal Y  : ---- Mean & Std: {0} & {1} \n'.format(
        np.array(result['optimalYs']).mean(), np.array(result['optimalYs']).std()))
    print('Runtime    : ---- Mean & Std: {0} & {1} \n'.format(
        np.array(result['Runtimes']).mean(), np.array(result['Runtimes']).std()))
    print('FE Runtime : ---- Mean & Std: {0} & {1} \n'.format(
        np.array(result['FuncEvalRuntimes']).mean(), np.array(result['FuncEvalRuntimes']).std()))
    print('FE Ratio   : ---- Mean & Std: {0} & {1} \n'.format(
        np.array(FuncEvalRatio).mean(), np.array(FuncEvalRatio).std()))
    print('FE Number  : ---- Mean & Std: {0} & {1} \n'.format(
        np.array(result['FuncEvalNums']).mean(), np.array(result['FuncEvalNums']).std()))
    