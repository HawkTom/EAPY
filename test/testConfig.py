# !/usr/bin/env python
"""
testConfig.py

test the feasibility of input parameters

command in the terminal: nosetests -s -v eapy.test.testConfig

"""

from eapy.config.functionConfig import ContinueFunctionParameter
from eapy.config.algorithmConfig import AlgorithmParameter
from eapy.config.trialConfig import TrialParameter


class test_Config:

    @classmethod
    def setUpClass(cls):
        print('------------test start-------------')

    @classmethod
    def tearDownClass(xls):
        print('------------test finished-----------\n')

    def test_functionParameter_case1(self):
        ContinueFunctionParameter('Sphere', 30, -200, 100)

    def test_functionParameter_case2(self):
        ContinueFunctionParameter('Sphere', 30, -200, 100)

    def test_trialParameter_case1(self):
        TrialParameter(30, True, 20180408)

    def test_trialParameter_case2(self):
        TrialParameter(10, False, 20180410)

    def test_algorithmConfig_case1(self):
        AlgorithmParameter('CLPSO', 10000)

    def test_algorithmConfig_case2(self):
        AlgorithmParameter('wPSO', 20000)

