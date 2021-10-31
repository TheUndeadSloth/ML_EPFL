def build_poly(txorg,degree):
   
    polFunc = np.ones([txorg.shape[0],txorg.shape[1]*(degree)+1])
    for j in range(degree):
        #trying with mean now
        polFunc[:,1+30*j:1+30*j+30] = np.power(txorg,j+1)
    return polFunc