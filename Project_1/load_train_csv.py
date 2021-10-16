import numpy as np
def load_train():
    traincsv = np.loadtxt('Data/train.csv',delimiter=',',skiprows=1,unpack=True)
    x = traincsv[2:,:]
    y = traincsv[1,:]
    return x,y 