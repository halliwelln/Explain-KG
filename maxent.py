#!/usr/bin/env python3

import sys
import numpy as np
import scipy.sparse as sparse
from scipy.optimize import minimize
from scipy.optimize import check_grad

class BGDistr(object):

    def __init__(self, *args, **kwargs):
        super(BGDistr, self).__init__()

        self.A = args[0]
        self.row_sum = np.sum(self.A, axis=0)

        self.row_val, self.row_idx, self.row_freq = np.unique(self.row_sum, axis=1, return_counts=True, return_inverse=True)

        self.n = self.row_val.shape[1]
        self.MN = self.row_freq * self.row_freq[:, None]

    def lagrangian(self, x):
        E = np.exp(x/2 + x[:,None]/2)
        M = np.log(1 + E)
        diag_M = np.diag(M)*self.row_freq
        M *= self.MN
        return np.sum(M) - np.sum(diag_M) - np.sum(x*self.row_freq*self.row_val)

    def gradient(self, x):
        E = np.exp(x/2 + x[:,None]/2)
        M = E / (1 + E)
        diag_M = np.diag(M)*self.row_freq
        M *= self.MN
        return np.sum(M, axis=0) - diag_M - self.row_freq*self.row_val


    def fit(self, verbose=True, tol=1e-5, iterations=100, **kwargs):
        x0 = np.random.randn(self.n)
        res = minimize(self.lagrangian, x0,
                       method='L-BFGS-B',
                       jac=self.gradient,
                       options={'disp': verbose, 'maxcor': 20, 'ftol': 1e-40,
                                'gtol': tol, 'maxiter': iterations, 'maxls': 20})
        self.la = res.x

    def get_row_probability(self, row_id, col_ids):
        '''
        Compute prior (degree) probability for the entries in a row specified
        by row_id.
        '''
        row_la = self.la[self.row_idx[row_id]]
        col_las = self.la[self.row_idx[col_ids]]

        E = np.exp(row_la/2 + col_las/2)
        P_i = E/(1+E)

        if row_id in col_ids:
            P_i[col_ids.index(row_id)] = 0 + sys.float_info.epsilon

        return P_i

    @property
    def parameters(self):
        return {'lambdas': self.la/2, 'indexer': self.row_idx}