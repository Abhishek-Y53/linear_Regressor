import numpy as np

class Linear_regressor_fit:
    def __init__(self,x,y):
        self.x=np.c_[np.ones((x.shape[0],1)),x]
        self.y=y
        self.coefficient=None

    def fit_fun(self):
        x_transpose=np.transpose(self.x)
        x_transpose_x=np.dot(x_transpose,self.x)
        x_transpose_y=np.dot(x_transpose,self.y)
        xt_x_inverse=np.linalg.inv(x_transpose_x)
        self.coefficient=np.dot(xt_x_inverse,x_transpose_y)

    def predict(self,x_test):
        x_test=np.c_[np.ones((x_test.shape[0],1)),x_test]
        return np.dot(x_test,self.coefficient)