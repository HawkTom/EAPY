#!/usr/bin/env python
"""
trailConfig.py

Description:  setting parameters for several runs of algorithm
              if you only need to run one time, no need to config these parameters

Parameters: 

    testNumber:     the total number for runing algorithm
    testLog:        determine whether print the result information or not, true default
    testSeed:       the seed for random generator, current date default
    testSeeds:      different seed for each run, no need to setting, autometally setting for each run


Eample:

    from trialConfig import *
    tp1 = TrialParameter(30)
    tp2 = TrialParameter(30, False)
    tp3 = TrialParameter(30, False, 123456)

"""

import numpy as np
import datetime

__all__ = ['TrailParameter']


class TrialParameter:

    def __init__(self, testNumber=30, testLog=True, testSeed=int(datetime.date.today().strftime("%Y%m%d"))):
        assert isinstance(testNumber, int), 'test number format need to be int'
        assert testNumber > 0, 'the total number of test must be greater than 0'
        assert isinstance(testLog, bool), 'test seed format need to be boolean'
        assert testSeed > 0, 'the random seed must be greater than or equal to 0'
        assert isinstance(testSeed, (int, float)
                          ), 'test seed format need to be numerical'

        self.testNumber = testNumber
        self.testLog = testLog
        self.testSeed = testSeed
        self.testSeeds = testSeed + 2 * np.array(range(testNumber))
