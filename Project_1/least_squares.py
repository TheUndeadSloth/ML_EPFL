import numpy as np
def least_squares(y, tx):
    
    """calculate the least squares solution."""
    
    w = np.linalg.solve((np.transpose(tx) @ tx), tx.T@y )
   
    e = y-tx @ w
    mse  = 1/(2*len(y))*e.T@e
  
    return mse[0,0], w
