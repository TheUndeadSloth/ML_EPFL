from implemenations import *
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
def cross_validation_visualization(lambds, mse_tr, mse_te,name,degree):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx    (lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda^%s"%degree)
    plt.ylabel("rmse")
  
    plt.title("cross validation "+name)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("CD_"+name)
def compute_mse(y,tx,w):
    e = y - tx @ w
    mse =  1/(2*len(y))*e.T@e
    return mse

def cross_validation(y,tx, degree):
    w_init = least_squares(y,tx)[1]
    w_init = np.zeros(w_init.shape)
    #w_init = np.zeros(w_init.shape)
    #gradient descent -15 to -6.5 gives good overview gamma = 3.727593720314938e-07 with degree 5
    #logistic gradeint diverges after around -14 graph until 13.8 gives good overview -14 good with rmse
    lambdas = np.logspace(-10, -3,30)
    rmse_te = []
    rmse_tr = []
    min_loss1 = 100000000000000000
    min_lambda = 0
    for lambda_ in lambdas:
        k_indices = build_k_indices(y,4,5)
        tr = []
        te = []
        
        for k in range(len(k_indices)):

            

            #make a mask to extract all test data
            mask = np.zeros(tx.shape[0], dtype=bool)
            mask[k_indices[k]] = True

            train_x = tx[mask,...]
            train_y = y[mask]
            amask = np.invert(mask)
            test_x = tx[amask,...]
            test_y = y[amask]
            """TODO add function you want to evaluate"""
            raise NotImplementedError

            # calculate the loss for train and test data:
            """rmse loss"""
            loss_tr = np.sqrt(2 * compute_mse(train_y, train_x , weights))
            loss_te = np.sqrt(2 * compute_mse(test_y, test_x , weights))
        
            """logistic loss"""
            
            # loss_tr = calculate_loss(train_y, train_x, weights)
            # loss_te = calculate_loss(test_y, test_x, weights)
            """predictive loss"""
            # e = train_y - predict_labels(weights,train_x)
            # loss_tr = np.sqrt(1/len(e)*e.T @ e)
            # e = test_y - predict_labels(weights,test_x)
            # loss_te = np.sqrt(1/len(e)*e.T @ e)
            tr.append(loss_tr)
            te.append(loss_te)

        rmse_tr.append(np.array(tr).mean())
        rmse_te.append(np.array(te).mean())
        
        
        if(rmse_te[-1]<min_loss1):
            min_lambda = lambda_
            min_loss1 = rmse_te[-1]
        
            
    degreeStr = str(1)
    cross_validation_visualization(lambdas, rmse_tr, rmse_te, 'GD test degree = %s' % degree,degreeStr)
    print(min_lambda)
    print(min_loss1)