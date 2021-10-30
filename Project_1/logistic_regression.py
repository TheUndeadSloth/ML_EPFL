import numpy as np
def logistic_regression(y, tx, initial_w,max_iters, gamma):

    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    w = initial_w
    for i in range(max_iters):
        gradient = calculate_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = calculate_loss(y, tx , w)
    return loss, w
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sig = np.exp(tx @ w)/(1+np.exp(tx @ w))
    grad = tx.T @ (sig - y)
    return grad
def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    
    loss = np.sum(-y*(tx @ w) + np.log(1+np.exp(tx @ w)))
    
 
    return loss

def penalized_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    """return the loss, gradient"""
    w = initial_w
    for i in range(max_iters):
        gradient = calculate_gradient(y, tx , w) + 2*lambda_*w
        w = w - gamma*gradient

    loss = calculate_loss(y, tx , w) + lambda_*w.T @ w
    return loss, gradient
def compute_loss_log_reg(y,tx,w):
    """Computes loss for logistic regression"""
    e = y - sigmoid(tx @ w)
    mse =  1/(2*len(y))*e.T@e
    return mse
def sigmoid(t):
    """apply the sigmoid function on t."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    return np.exp(t)/(1+np.exp(t))