import numpy as np
def learning_by_newton_method(y, tx, w, gamma, max_iters):
    for i in range(max_iters):
        loss, gradient, H = logistic_regression(y, tx, w)
        w = np.linalg.solve(H,-gradient+H @ w*gamma)
    return loss, w