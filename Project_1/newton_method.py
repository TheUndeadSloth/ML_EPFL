import numpy as np
def learning_by_newton_method(y, tx, w, gamma, max_iters):
    for i in range(max_iters):
        loss, gradient, H = logistic_regression(y, tx, w)
        w = np.linalg.solve(H,-gradient+H @ w*gamma)
    return loss, w
def logistic_regression(y, tx, w):
    """return the loss, gradient, and Hessian."""
    gradient = calculate_gradient(y, tx ,w)
    loss = calculate_loss(y, tx ,w)
    H = calculate_hessian(y, tx , w)
    return loss, gradient, H
def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    sn = []
    for ind in range(len(y)):
        element = sigmoid(tx[ind,:].T @ w) *(1-sigmoid(tx[ind,:].T @ w))
        sn.append(element[0])
    
    sn = np.array(sn)
    S = np.diag(sn)
    
    H = tx.T @ S @ tx
    return H
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sig = np.exp(tx @ w)/(1+np.exp(tx @ w))
    grad = tx.T @ (sig - y)
    return grad