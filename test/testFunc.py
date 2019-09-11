# !/usr/bin/env python
"""
testFunc.py

test the benchmark function at first.

1. You can test all function at a time.

   Command in the terminal:  nosetests -s -v eapy.test.testFunc

2. You can test one function one time.
   
   Example: nosetests -s -v eapy.test.testFunc.test_Sphere

"""


from eapy.function import continueFunction as cF
import numpy as np
import nose


class test_Sphere():

    @ classmethod
    def setUpClass(clf):
        print('--Sphere function point test start--')

    @ classmethod
    def tearDownClass(clf):
        print('--Sphere function point test finish--\n')

    def test_case1(self):
        point = np.zeros((3, 7))
        value = np.array([0, 0, 0])
        # assert 2 == 3
        mask = np.equal(value, cF.Sphere(point)).all()
        nose.tools.ok_(mask, msg='Sphere function is incorrect')

    def test_case2(self):
        point = np.array([[3, 9, 19, 16, 20]])
        value = 1107
        nose.tools.eq_(value, cF.Sphere(point),
                       msg='Sphere function is incorrect')

    def test_case3(self):
        point = 2 * np.ones((3, 7))
        value = np.array([28, 28, 28])
        mask = np.equal(value, cF.Sphere(point)).all()
        nose.tools.ok_(mask, msg='Sphere function is incorrect')

    def test_case4(self):
        point = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0],
                          [-1.0, -1.0, -1.0, -1.0, -1.0],
                          [1.0, -1.0, 1.0, -1.0, 1.0],
                          [1.0, 2.0, 3.0, 4.0, 5.0],
                          [1.0, -2.0, 3.0, -4.0, 5.0],
                          [5.0, 4.0, 3.0, 2.0, 0.0],
                          [1.1, 1.2, 1.3, 1.4, -1.5]])
        value = np.array([0, 5.0000, 5.0000, 5.0000,
                          55.0000, 55.0000, 54.0000, 8.5500])
        mask = np.equal(value, cF.Sphere(point)).all()
        nose.tools.ok_(mask, msg='Sphere function is incorrect')


class test_Rastrigin():

    @ classmethod
    def setUpClass(clf):
        print('--Rastrigin function point test start--')

    @ classmethod
    def tearDownClass(clf):
        print('--Rastrigin function point test finish--\n')

    def test_case1(self):
        point = np.zeros((3, 7))
        value = np.array([0, 0, 0])
        mask = np.equal(value, cF.Rastrigin(point)).all()
        nose.tools.ok_(mask, msg='Rastrigin function is incorrect')
        # 0, 0, 0

    def test_case2(self):
        point = np.array([[3, 9, 19, 16, 20]])
        value = 1107
        nose.tools.eq_(value, cF.Rastrigin(point),
                       msg='Rastrigin function is incorrect')
        # 1107

    def test_case3(self):
        point = 2 * np.ones((3, 7))
        value = np.array([28, 28, 28])
        mask = np.equal(value, cF.Rastrigin(point)).all()
        nose.tools.ok_(mask, msg='Rastrigin function is incorrect')
        # 28, 28, 28

    def test_case4(self):
        point = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0],
                          [-1.0, -1.0, -1.0, -1.0, -1.0],
                          [1.0, -1.0, 1.0, -1.0, 1.0],
                          [1.0, 2.0, 3.0, 4.0, 5.0],
                          [1.0, -2.0, 3.0, -4.0, 5.0],
                          [5.0, 4.0, 3.0, 2.0, 0.0],
                          [1.1, 1.2, 1.3, 1.4, -1.5]])
        value = np.array([0, 5.0000, 5.0000, 5.0000,
                          55.0000, 55.0000, 54.0000, 68.5500])
        mask = np.equal(value, cF.Rastrigin(point)).all()
        nose.tools.ok_(mask, msg='Rastrigin function is incorrect')
        # 0, 5.0000, 5.0000, 5.0000, 55.0000, 55.0000, 54.0000, 68.5500


class test_Griewank():

    @ classmethod
    def setUpClass(clf):
        print('--Griewank function point test start--')

    @ classmethod
    def tearDownClass(clf):
        print('--Griewank function point test finish--\n')

    def test_case1(self):
        value = np.array([0, 0, 0])
        point = np.zeros((3, 7))
        mask = np.equal(value, cF.Griewank(point)).all()
        nose.tools.ok_(mask, msg='Griewank function is incorrect')
        # 0, 0, 0

    def test_case2(self):
        point = np.array([[3, 9, 19, 16, 20]])
        value = 1.2735
        nose.tools.eq_(value, np.round(cF.Griewank(point), 4),
                       msg='Griewank function is incorrect')
        # 1.2735

    def test_case3(self):
        point = 2 * np.ones((3, 7))
        value = np.array([1.0114, 1.0114, 1.0114])
        mask = np.equal(value, np.round(cF.Griewank(point), 4)).all()
        nose.tools.ok_(mask, msg='Griewank function is incorrect')
        # 1.0114, 1.0114, 1.0114

    def test_case4(self):
        point = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0],
                          [-1.0, -1.0, -1.0, -1.0, -1.0],
                          [1.0, -1.0, 1.0, -1.0, 1.0],
                          [1.0, 2.0, 3.0, 4.0, 5.0],
                          [1.0, -2.0, 3.0, -4.0, 5.0],
                          [5.0, 4.0, 3.0, 2.0, 0.0],
                          [1.1, 1.2, 1.3, 1.4, -1.5]])
        value = np.array([0, 0.7289, 0.7289, 0.7289,
                          1.0172, 1.0172, 0.9901, 0.8708])
        mask = np.equal(value, np.round(cF.Griewank(point), 4)).all()
        nose.tools.ok_(mask, msg='Griewank function is incorrect')
        # 0, 0.7289, 0.7289, 0.7289, 1.0172, 1.0172, 0.9901, 0.8708


class test_Rosenbrock():

    @ classmethod
    def setUpClass(clf):
        print('--Rosenbrock function point test start--')

    @ classmethod
    def tearDownClass(clf):
        print('--Rosenbrock function point test finish--\n')

    def test_case1(self):
        point = np.zeros((3, 7))
        value = np.array([6, 6, 6])
        mask = np.equal(value, cF.Rosenbrock(point)).all()
        nose.tools.ok_(mask, msg='Rosenbrock function is incorrect')
        # 6, 6, 6

    def test_case2(self):
        point = np.array([[3, 9, 19, 16, 20]])
        value = 17857117
        nose.tools.eq_(value, cF.Rosenbrock(point),
                       msg='Rosenbrock function is incorrect')
        # 17857117

    def test_case3(self):
        point = 2 * np.ones((3, 7))
        value = np.array([2406, 2406, 2406])
        mask = np.equal(value, cF.Rosenbrock(point)).all()
        nose.tools.ok_(mask, msg='Rosenbrock function is incorrect')
        # 2406, 2406, 2406

    def test_case4(self):
        point = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0],
                          [-1.0, -1.0, -1.0, -1.0, -1.0],
                          [1.0, -1.0, 1.0, -1.0, 1.0],
                          [1.0, 2.0, 3.0, 4.0, 5.0],
                          [1.0, -2.0, 3.0, -4.0, 5.0],
                          [5.0, 4.0, 3.0, 2.0, 0.0],
                          [1.1, 1.2, 1.3, 1.4, -1.5]])
        value = np.array([4, 0, 1.616e+3, 8.08e+2, 1.4814e+4,
                          3.0038e+4, 6.7530e+4, 1.20784e+3])
        mask = np.equal(value, np.round(cF.Rosenbrock(point), 4)).all()
        nose.tools.ok_(mask, msg='Rosenbrock function is incorrect')
        # 4, 0, 1.616e+3, 8.08e2, 1.4814e+4,3.0038e+4, 6.7530e+4, 1.2078e+3


class test_Schwefel12():

    @ classmethod
    def setUpClass(clf):
        print('--Schwefel12 function point test start--')

    @ classmethod
    def tearDownClass(clf):
        print('--Schwefel12 function point test finish--\n')

    def test_case1(self):
        point = np.zeros((3, 7))
        value = np.array([0, 0, 0])
        mask = np.equal(value, cF.Schwefel12(point)).all()
        nose.tools.ok_(mask, msg='Schwefel12 function is incorrect')
        # 0, 0, 0

    def test_case2(self):
        point = np.array([[3, 9, 19, 16, 20]])
        value = 7812
        nose.tools.eq_(value, cF.Schwefel12(point),
                       msg='Schwefel12 function is incorrect')
        # 7812

    def test_case3(self):
        point = 2 * np.ones((3, 7))
        value = np.array([560, 560, 560])
        mask = np.equal(value, cF.Schwefel12(point)).all()
        nose.tools.ok_(mask, msg='Schwefel12 function is incorrect')
        # 560, 560, 560

    def test_case4(self):
        point = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0],
                          [-1.0, -1.0, -1.0, -1.0, -1.0],
                          [1.0, -1.0, 1.0, -1.0, 1.0],
                          [1.0, 2.0, 3.0, 4.0, 5.0],
                          [1.0, -2.0, 3.0, -4.0, 5.0],
                          [5.0, 4.0, 3.0, 2.0, 0.0],
                          [1.1, 1.2, 1.3, 1.4, -1.5]])
        value = np.array([0, 55, 55, 3, 371, 19, 642, 56.71])
        mask = np.equal(value, np.round(cF.Schwefel12(point), 4)).all()
        nose.tools.ok_(mask, msg='Schwefel12 function is incorrect')
        # 0, 55, 55, 3, 371, 19, 642, 56.71
