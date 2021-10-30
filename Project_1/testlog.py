import numpy as np
def logistic_regression_test(y, tx, init_w,max_iters,gamma):

	"""Logistic regression using gradient descent."""
	iter = 0
	
	N = tx.shape[0]
	w = init_w
	y_zero = y
	y_zero[y == -1] = 0
	
	for i in range(max_iters):
		h = sigmoid(np.matmul(tx, w))
		grad = np.matmul(tx.T, (h - y)) / N

		w = w - gamma * grad

		iter += 1
		

	loss = compute_loss_log_reg(y,tx, w)
	print('with loss {l:.8f}.'.format(l=loss))

	return w, loss

def compute_loss_log_reg(y,tx, w):
    """compute the loss: negative log likelihood."""
    
    loss = np.sum(-y*(tx @ w) + np.log(1+np.exp(tx @ w)))
    
 
    return loss
	
def sigmoid(t):
    """apply the sigmoid function on t."""
    return np.exp(t)/(1+np.exp(t))