import pandas as pd
import numpy as np

if __name__=="__main__":
    dataset=pd.read_csv('Data.csv')
    x=dataset.iloc[:, :-1].values
    y=dataset.iloc[:-1].values
    