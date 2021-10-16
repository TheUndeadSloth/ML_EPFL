def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    w = initial_w
    for n_iter in range(max_iters):
        e = y -tx @ np.transpose(w)
        loss = 1/(len(e))*e.dot(e)
        gradient = compute_stoch_gradient(y,tx,w)
        w = w-gradient
       
       
    return loss, w