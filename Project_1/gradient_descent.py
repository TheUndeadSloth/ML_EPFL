import numpy as np
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    
    w = initial_w
    for n_iter in range(max_iters):
        e = y -tx @ w
        
        mse  = 1/(2*len(y))*e.T@e
        gradient = -1/(len(e))*(tx.T @ e)
        
        w = w-gradient*gamma
       
 

    return mse, w