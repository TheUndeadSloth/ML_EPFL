import numpy as np

"""
every function returns (w, loss)
"""
def least_squares(y, tx):
    
    
    """calculate the least squares solution.""" 
    w = np.linalg.solve((np.transpose(tx) @ tx), tx.T@y )
    e = y-tx @ w
    mse  = 1/(2*len(y))*e.T@e
    return w, mse[0,0]

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    
    w = initial_w
    for n_iter in range(max_iters):
        e = y -tx @ w
        
        mse  = 1/(2*len(y))*e.T@e
        gradient = -1/(len(e))*(tx.T @ e)
        
        w = w-gradient*gamma
       
 

    return w, mse
def ridge_regression(y, tx, lambda_):
    x2 =  np.transpose(tx) @ tx
   
    w = np.linalg.solve(x2 + 2*len(y)*lambda_*np.identity(x2.shape[0]),np.transpose(tx) @ y)
    e = y-tx @ w
    mse =  1/(2*len(y))*e.T@e
    return w, mse



# ridge_regression(y, tx, lambda_):

#     return w, loss

# logistic_regression(y, tx, lambda_):

#     return w, loss

# reg_logistic_regression(y, tx, initial_w, max_iters, gamma):

#     return w, loss