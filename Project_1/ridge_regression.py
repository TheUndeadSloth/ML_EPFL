# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np

def ridge_regression(y, tx, lambda_):
    x2 =  np.transpose(tx) @ tx
   
    w = np.linalg.solve(x2 + 2*len(y)*lambda_*np.identity(x2.shape[0]),np.transpose(tx) @ y)
    e = y-tx @ w
    mse =  1/(2*len(y))*e.T@e
    return mse, w