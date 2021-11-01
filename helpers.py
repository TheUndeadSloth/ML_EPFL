import numpy as np
import csv
import matplotlib.pyplot as plt

def load_data():
    y, txorg, ids = load_csv_data('Data/train.csv')
    y = np.array([y])
    y = y.T
    txTestorg, idsTest = load_csv_Test('Data/test.csv')
    return y, txorg, txTestorg, idsTest

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]
    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = 0

    return yb, input_data, ids

def load_csv_Test(data_path):
    """Loads data and returns tX (features) and ids (event ids)"""
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]
    return input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission
    Arguments:  ids (event ids associated with each prediction)
                y_pred (predicted class labels)
                name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def generetate_csv_prediction(idsTest, w, txTest, filename):
    """Takes the weight and generates a .csv prediction"""
    yPred = predict_labels(w, txTest)

    with open(filename, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(idsTest, yPred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = data @ weights
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Batch_iter function from lab 2"""
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def build_k_indices(y, k_fold, seed):
    """Build k indices from lab 4"""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(txorg, degree):
    """Function to make polynomial expansion of original tx data"""
    polFunc = np.ones([txorg.shape[0],txorg.shape[1]*(degree)+1])
    for j in range(degree):
        polFunc[:,1+30*j:1+30*j+30] = np.power(txorg,j+1)
    return polFunc

def compute_mse(y,tx,w):
    e = y - tx @ w
    mse =  1/(2*len(y))*e.T@e
    return mse

def sigmoid(t):
    return np.exp(t)/(1 + np.exp(t))