import numpy as np
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss

    w = initial_w
    for n_iter in range(max_iters):
        e = y-tx @ np.transpose(w)
        
        loss = 1/(len(e))*e @ np.transpose(e)
        print(loss)
        gradient = -1/(len(e))*(np.transpose(tx) @ e)
        w = w-gradient*gamma
       
        
        

    return loss, w

def gradient_descentE(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
  
    w = initial_w
    for n_iter in range(max_iters):
        e = y -tx @ np.transpose(w)
        loss = 1/(len(e))*e @ np.transpose(e)
        gradient = -1/(len(e))*(np.transpose(tx) @ e)
        
        w = w-gradient*gamma
       
       

    return loss, w