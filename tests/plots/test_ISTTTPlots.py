__author__ = 'lei'
import unittest
import python.plots.ISTTTPlots as ISTTTPlots
import numpy as np

class TestExperimentResults(unittest.TestCase):
    @staticmethod
    def make_fake_data():
        A = np.ones((5,5))
        x_true = np.ones((5,))
        b = np.dot(A, x_true)
        U = np.eye(5)
        x = .5 * np.ones((5,))
        f = np.ones((5,)) * 3
        return A, b, x, x_true, U, f

    @staticmethod
    def make_ExperimentResults():
        A, b, x, x_true, U, f = TestExperimentResults.make_fake_data()
        return ISTTTPlots.ExperimentResults(A, b, x, x_true, U, f, None, 0, 0)

    def test_flow_error(self):
        er = TestExperimentResults.make_ExperimentResults()
        self.assertAlmostEqual(er.flow_error(), 7.5/15)

    def test_flow_error2(self):
        er = TestExperimentResults.make_ExperimentResults()
        er.x[0] = 2
        self.assertAlmostEqual(er.flow_error(), 9.0/15)

    def test_geh(self):
        er = TestExperimentResults.make_ExperimentResults()
        for x,y in zip(er.geh(), 5.0/3 * np.ones((5,))):
            self.assertAlmostEqual(x*x, y)

    def test_geh2(self):
        er = TestExperimentResults.make_ExperimentResults()
        er.b[0] = 2.5
        for x,y in zip(er.geh(), [0,5.0/3,5.0/3,5.0/3,5.0/3]):
            self.assertAlmostEqual(x*x, y)

    def test_nullity(self):
        er = TestExperimentResults.make_ExperimentResults()
        #self.assertAlmostEqual(er.nullity(), 0)

class TestFilterFunctions(unittest.TestCase):
    def test_between(self):
        self.assertTrue(ISTTTPlots.between(1,0,3))
        self.assertFalse(ISTTTPlots.between(-1,0,3))
        self.assertFalse(ISTTTPlots.between(10,0,3))

    def test_select(self):
        l1 = [1,2,3,4]
        l2 = [1,2,3,4]
        r1, r2 = ISTTTPlots.select(l1, l2, lambda x: x <= 2)
        self.assertSequenceEqual(r1, [1,2])
        self.assertSequenceEqual(r2, [1,2])

class TestSliceFunction(unittest.TestCase):
    def test_slice_zero(self):
        xs = [0, 5, 6]
        ys = [1, 2, 3]
        y, xs, ys = ISTTTPlots.slice_zero(xs, ys)
        self.assertSequenceEqual(xs, [5, 6])
        self.assertSequenceEqual(ys, [2, 3])
        self.assertEqual(y, 1)

if __name__ == '__main__':
    unittest.main()