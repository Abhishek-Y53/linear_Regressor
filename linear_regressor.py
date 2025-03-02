import pandas as pd
import numpy as np
from linear_regerssor_fit import Linear_regressor_fit
from encoder import Encoder
from feature_scaling import Feature_scaling

if __name__=="__main__":
    dataset=pd.read_csv('data.csv')
    # print(dataset.head)
    # print(dataset.shape)
    x=dataset.iloc[:, :-1].values
    y=dataset.iloc[:,3].values
    normalize=Feature_scaling()
    x=normalize.normalization(x,0,1)
    print(x)
    encode=Encoder()
    X=encode.encoder(x,2).astype(float)
    X=X.astype(float)
    print(X)
    print(x.shape)
    regressor=Linear_regressor_fit(X,y)
    regressor.fit_fun()
    prediction=regressor.predict(np.array([[500,3,1,0,0]]))
    print("Predictions: ",prediction)
