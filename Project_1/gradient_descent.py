def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        e = y -tx @ np.transpose(w)
        loss = 1/(len(e))*e @ np.transpose(e)
        gradient = -1/(len(e))*(np.transpose(tx) @ e)
        
        w = w-gradient*1.3
       
        
        

    return loss, w