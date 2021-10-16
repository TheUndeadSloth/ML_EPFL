def least_squares(y, tx):
    
    """calculate the least squares solution."""
   
    w = np.linalg.solve((np.transpose(tx) @ tx), np.transpose(tx) @ y)
    print(w)
    e = y-tx @ w
    mse  = 1/(2*len(y))*e@np.transpose(e)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    return mse, w

