__author__ = 'lei'

import scipy.io as sio
import util

class ExperimentProblem:
    def __init__(self, problem_path):
        A, b, N, block_sizes, x_true, nz, f = util.load_data(problem_path)

        self._A = A
        self._b = b
        self._U = util.U(block_sizes)
        self._x_true = x_true
        self._f = f
        self._N = N
        self._x0 = util.block_sizes_to_x0(block_sizes)

class ExperimentSolution:
    def __init__(self, solution_path):
        sol = sio.loadmat(solution_path)
        self._z = sol['x']

class Experiment:
    def __init__(self, problem_path, solution_path):
        self.problem = ExperimentProblem(problem_path)
        self.solution = ExperimentSolution(solution_path)

    def A(self):
        return self.problem._A

    def b(self):
        return self.problem._b

    def U(self):
        return self.problem._U

    def x_true(self):
        return self.problem._x_true

    def x_est(self):
        return self.problem._N *self.solution._z +self.problem._x0
