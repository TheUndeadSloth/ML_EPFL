import numpy as np
from implemenations import *
from helpers import *
from Additional_Functions import compute_mse

if __name__ == "__main__":
    #code here
    y, txTemp, txTestTemp, idsTest = load_data()
   
    tx = build_poly(txTemp,1)
    txTest = build_poly(txTestTemp,1)

    w_init = np.zeros(tx.shape[1])
    
    #remember to output prediction
    """Least squares prediction:"""
    w_ls, mse = least_squares(y, tx)
    generetate_csv_prediction(idsTest, w_ls, txTest, "testSub_least_squares.csv")
    print(mse)

    """Gradien descent prediction. Got gamma with the help of cross validation, see additional functions"""
    w_GD, mse = least_squares_GD(y, tx, w_init, 500, 2.395026619987486e-07)
    generetate_csv_prediction(idsTest, w_GD, txTest, "testSub_Gradient_Descent.csv")
    print(mse)

    """Ridge regresion prediction, lambda decided in same way as gamma above"""
    w_RR, mse = ridge_regression(y, tx, 3.290344562312671e-06)
    generetate_csv_prediction(idsTest, w_RR, txTest, "testSub_Ridge_Regression.csv")
    print(mse)

    """Stochastic gradient descent prediction. Decided to use similar gamma as in gradient descent"""
    w_SGD, mse = least_squares_SGD(y, tx, w_init, 500, 2.395026619987486e-07)
    generetate_csv_prediction(idsTest, w_SGD, txTest, "testSub_Stochastic_Gradient_Descent.csv")
    print(mse)

    """Logistic regression prediction. Got gamma with the help of cross validation, see additional functions"""
    w_LR, log_loss = logistic_regression(y, tx, w_init, 500, 1e-15)
    mse = compute_mse(y, tx, w_LR)
    generetate_csv_prediction(idsTest, w_LR, txTest, "testSub_Logistic_Regression.csv")
    print(mse)
    print(log_loss)

    """Regularized logistic regression prediction. Got lambda from cross validation and used same gamma as in logistic regression"""
    w_RLR, log_loss = reg_logistic_regression(y, tx, 1e5, w_init, 500, 1e-15)
    mse = compute_mse(y, tx, w_RLR)
    generetate_csv_prediction(idsTest, w_RLR, txTest, "testSub_Regularized_Logistic_Regression.csv")
    print(mse)
    print(log_loss)

    """The best test result we got"""
    tx1 = build_poly(txTemp,5)
    txTest1 = build_poly(txTestTemp,5)
    w_Best, mse = ridge_regression(y, tx1, 0.11)
    generetate_csv_prediction(idsTest, w_Best, txTest1, "testSub_Best.csv")
    print(mse)