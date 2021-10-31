import numpy as np
from implemenations import *
from helpers import *

if __name__ == "__main__":
    #code here
    y, txTemp, txTestTemp, idsTest = load_data()
   
    tx = build_poly(txTemp,1)
    txTest = build_poly(txTestTemp,1)
    
    #remember to output prediction
    """Least squares prediction:"""
    w_least_squares, mse = least_squares(y,tx)
    generetate_csv_prediction(idsTest, w_least_squares, txTest, "testSub_least_squares.csv")
    print(mse)
    w_init = np.zeros(w_least_squares.shape)
    """Gradien descent prediction, gamma gotten with the help of cross validation see additional functions"""
    w_GD, mse = least_squares_GD(y,tx, w_init,500,2.395026619987486e-07)
    generetate_csv_prediction(idsTest, w_GD, txTest, "testSub_Gradient_Descent.csv")
    print(mse)
    """Ridge regresion prediction, lambda decided in same way as gamma"""
    w_RR, mse = ridge_regression(y, tx, 3.290344562312671e-06)
    generetate_csv_prediction(idsTest, w_GD, txTest, "testSub_Ridge_Regression.csv")
    print(mse)
    """The  best test result we got"""
    tx1 = build_poly(txTemp,5)
    txTest1 = build_poly(txTestTemp,5)
    w_Best, mse = ridge_regression(y, tx1, 0.11)
    generetate_csv_prediction(idsTest, w_Best, txTest1, "testSub_Best.csv")
    print(mse)