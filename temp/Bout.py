#!/usr/bin/env python
"""
configuration for parameters of several testing 
"""
__authors__ = "Hao Tong"
__date__ = "2018/03/26"
__version__ = "0.0.1"

import datetime
import numpy as np

class TestParameter:

    def __init__(testNumber = 30, testLog = True, testSeed = int(datetime.date.today().strftime("%Y%m%d"))):
        if not isinstance(testNumber, int):
            raise NameError("test number format need to be int")
        if not isinstance(testLog, bool)
            raise NameError("test seed format need to be boolean")
        if not isinstance(testLog, (int, float))
            raise NameError("test seed format need to be numerical")
        self.testNumber = testNumber
        self.testLog = testLog
        self.testSeed = testSeed
        self.testSeeds = testSeed + 2 * np.array(range(testNumber))
