import numpy as np
from helpers import *
"""Least squares method"""
def least_squares(y, tx):
    w = np.linalg.solve((np.transpose(tx) @ tx), tx.T@y)
    mse = compute_mse(y, tx, w)

    return w, mse
"""Gradient descent optimising mse"""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        e = y -tx @ w

        gradient = -1/(len(e))*(tx.T @ e)
        w = w-gradient*gamma

    mse = compute_mse(y, tx, w)

    return w, mse


def ridge_regression(y, tx, lambda_):
    x2 =  np.transpose(tx) @ tx
   
    w = np.linalg.solve(x2 + 2*len(y)*lambda_*np.identity(x2.shape[0]),np.transpose(tx) @ y)
    mse = compute_mse(y, tx, w)

    return w, mse

"""Stochastic gradient descent optimising mse"""
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for i in range(max_iters):

        #setup for computing gradient
        gradient = np.empty(tx.shape[1])

        #Using 1 as batch size
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            """Compute a gradient from just few examples n and their corresponding y_n labels."""
            #using derivative of mse
            eff = (w * minibatch_tx).sum(axis=1)
            e = minibatch_y - eff
            
            #make vector of e vector times sequenced columns of tx (because of the different derivatives sequence one tx each)
            for j in range(minibatch_tx.shape[1]):
                gradient[j] = (-(e * minibatch_tx[:,j]) * (1/(minibatch_tx.shape[0]))).sum()

            #update weight by gradient
            w = w - gamma * gradient
    log_loss = (np.log(1 + np.exp(tx @ w)) - y*(tx @ w)).sum()
    mse = compute_mse(y, tx, w)

    return w, mse, log_loss

"""Logistic gradient descent"""
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for i in range(max_iters):
        gradient = np.matmul(tx.T,(sigmoid(np.matmul(tx, w)) - y))
        w = w - gamma * gradient
    
    log_loss = (np.log(1 + np.exp(tx @ w)) - y*(tx @ w)).sum()

    return w, log_loss

"""Penalized Logistic gradient descent"""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w

    for i in range(max_iters):
        gradient = np.matmul(tx.T,(sigmoid(np.matmul(tx, w)) - y)) + lambda_ * w
        w = w - gamma * gradient
    
    w_loss = ((lambda_ / 2) * w).sum()
    log_loss = (np.log(1 + np.exp(tx @ w)) - y*(tx @ w)).sum() + w_loss

    return w, log_loss