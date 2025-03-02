import numpy as np

class Feature_scaling:
    def __init__(self):
        pass

    def normalization(self,x,index_s,index_l):
        for index_s in range(index_s,index_l+1):
            x_max=0
            x_min=4444
            length=0
            for v in x[:,index_s]:
                length= length + 1
                if v<x_min:
                    x_min=v
                if v>x_max:
                    x_max=v
            if x_max==x_min:
                x[:,index_s]=0
            else:
                for i in range(0, length):
                    val = x[i, index_s]
                    val = (val - x_min) / (x_max-x_min)
                    x[i, index_s] = val
        return x