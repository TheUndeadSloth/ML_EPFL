from implementations import *
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
"""Creates graph for cross validation and saves in file with the name inputed """
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
"""This is cross validation that was used during our testing"""
def cross_validation(y,tx, degree):
    w_init = least_squares(y,tx)[1]
    w_init = np.zeros(w_init.shape)
   
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

            test_x = tx[mask,...]
            test_y = y[mask]
        
            amask = np.invert(mask)
            train_x = tx[amask,...]
            train_y = y[amask]
            """TODO add function you want to evaluate"""
            raise NotImplementedError

            # calculate the loss for train and test data:
            """rmse loss"""
            loss_tr = np.sqrt(2 * compute_mse(train_y, train_x , weights))
            loss_te = np.sqrt(2 * compute_mse(test_y, test_x , weights))
        
            """logistic loss"""
            # loss_tr = calculate_loss(train_y, train_x, weights)
            # loss_te = calculate_loss(test_y, test_x, weights
            tr.append(loss_tr)
            te.append(loss_te)

        rmse_tr.append(np.array(tr).mean())
        rmse_te.append(np.array(te).mean())
        
        
        if(rmse_te[-1]<min_loss1):
            min_lambda = lambda_
            min_loss1 = rmse_te[-1]
        
            
    degreeStr = str(1)
    cross_validation_visualization(lambdas, rmse_tr, rmse_te, 'GD test degree = %s' % degree,degreeStr)
    #Prints out the most effective hyperparameter and the corresponding loss
    print(min_lambda)
    print(min_loss1)