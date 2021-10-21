import numpy as np
def least_squares(y, tx):
    print("hej")
    """calculate the least squares solution."""
   
    w = np.linalg.solve((tx.T @ tx), y @ tx)
    
    e = y-tx @ w
    mse  = 1/(2*len(y))*e@np.transpose(e)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    return mse, w

