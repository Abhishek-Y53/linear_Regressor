import numpy as np

class Encoder:
    def __init__(self):
        pass
    def encoder(self,x,index):
        ls=[]
        m=0
        for i in x[:, index ]:
            m=m+1
            if i not in ls:
                ls.append(i)
        n=len(ls)
        k=1
        # n=n-1
        arr=np.zeros((m,n))
        for i in range (n):
            element=ls[i]
            for j in range(m):
                if x[j,index]==element:
                    arr[j,i]=k
            k=k+1
        new_x=np.delete(x,index,axis=1)
        return np.concatenate((new_x.astype(float),arr.astype(float)),axis=1)


