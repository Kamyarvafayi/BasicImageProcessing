import numpy as np
import pandas as pd
from sklearn import datasets
irisdata = datasets.load_iris()
Irisdata = irisdata['data']

column = ['feature1','feature2','feature3','feature4']
index = np.arange(0,150,dtype = int)
DataFrame = pd.DataFrame(data = Irisdata, columns = column,index = index)

centers = []
k = 3
for i in range(k):
    centers.append(DataFrame.iloc[int(np.floor(150/k*i))])

zerolist = [0 for i in range(150)]

for i in range(k):
    DataFrame["Dist {}".format(i)] = zerolist
    for j in range(150):
       DataFrame["Dist {}".format(i)].loc[j] = float(np.linalg.norm(DataFrame.iloc[j][0:4]-centers[i]))  
DataFrame['index'] = zerolist
DataFrame['index'] = DataFrame.iloc[:,4:-1].min(axis=1)

# Manipulating Data
DataFrame2 = DataFrame.drop(labels = [0,1,2,3],axis = 0,inplace = False)
DataFrame2.loc[0] = DataFrame.loc[0]
DataFrame2.sort_index()
