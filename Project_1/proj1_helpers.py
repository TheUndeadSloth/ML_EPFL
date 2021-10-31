# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = 0
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def load_csv_Test(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""

    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

 
    return input_data, ids
def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred



def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def the_mean_function(data, outlier):
    data = data.transpose()
    means = np.average(data, weights=(data != outlier), axis=0)
    newData = np.where(data[0] != outlier, data[0], means[0])
    for i in range(1, data.shape[0]):
        newData = np.vstack([newData,np.where(data[i] != outlier, data[i], means[i])])
    return newData.transpose()

def cross_validation(matrix, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    data = matrix
    
    #make a mask to extract all test data
    mask = np.zeros(data.shape[0], dtype=bool)
    mask[k_indices[k]] = True
    
    test = data[mask,...]
    amask = np.invert(mask)
    train = data[amask,...]
    
    train_x = train[:,1:]
    test_x = test[:,1:]
    
    train_y = train[:,0]
    test_y = test[:,0]
    
    # ridge regression
    
    weights = ridge_regression(train_y,train_x,lambda_)[1]
    
    # calculate the loss for train and test data:
    
    loss_tr = np.sqrt(2 * costs.compute_mse(train_y, train_x , weights))
    loss_te = np.sqrt(2 * costs.compute_mse(test_y, test_x , weights))
    
    return loss_tr, loss_te
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
def compute_mse(y,tx,w):
    e = y - tx @ w
    mse =  1/(2*len(y))*e.T@e
    return mse
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

