
# coding: utf-8

# In[53]:


import pandas
import numpy as np
import matplotlib.pyplot as plt
from math import exp


# In[54]:


url = "iris.txt"
names = ["x1","x2","x3","x4","class"]
data = pandas.read_csv(url, names=names)


# In[249]:

arr_sigmoid = np.zeros(6000)
arr_iris = np.zeros((6000,4))
arr_class = np.zeros(6000)
tetha = np.zeros((6001,5))
tetha[0] = ([0.4,0.1,0.2,0.6,0.9])
deltatetha = np.zeros((6000,5))
arr_prediction = np.zeros(6000)
arr_error = np.zeros(6000)
arr_hfuction = np.zeros(6000)
alpha = 0.1
class SGD:
    ROWS_DATA = 100
    def save_data(self):
        for i in range (6000):
            baris_csv = i % 100
            for j in range (5):
                if j==4:
                    if data.iloc[baris_csv][4] == 'Iris-setosa':
                        arr_class[i] = 1
                    else:
                        arr_class[i] = 0
                else:
                    arr_iris[i][j] = data.iloc[baris_csv][j]
                    
    def h_function(self,i):
        
        return arr_iris[i][0]*tetha[i][0]+arr_iris[i][1]*tetha[i][1]+arr_iris[i][2]*tetha[i][2]+arr_iris[i][3]*tetha[i][3]+tetha[i][4]
    
    def sigmoid(self,i):
        arr_sigmoid[i] = 1/(1+exp(-self.h_function(i)))
    
    def prediction(self,i):
        if arr_sigmoid[i]<0.5:
            arr_prediction[i] = 0
        else:
            arr_prediction[i] = 1
    
    def error(self,i):
        arr_error[i] = (arr_class[i]-arr_sigmoid[i])*(arr_class[i]-arr_sigmoid[i])
        
    def deltatetha(self,i):
        for j in range (5):
            if j == 4:
                deltatetha[i][j] = 2*(arr_sigmoid[i]-arr_class[i])*(1-arr_sigmoid[i])*arr_sigmoid[i]*1
            else:
                deltatetha[i][j] = 2*(arr_sigmoid[i]-arr_class[i])*(1-arr_sigmoid[i])*arr_sigmoid[i]*arr_iris[i][j]
                
    def tetha_baru(self,i):
        for j in range (5):
            tetha[i+1][j] = tetha[i][j]-alpha*deltatetha[i][j]
            
    def main(self):
        self.save_data()
        for i in range (6000):
            self.h_function(i)
            self.prediction(i)
            self.sigmoid(i)
            self.error(i)
            self.deltatetha(i)
            self.tetha_baru(i)


# In[259]:


x = SGD()
x.main()
sum_error = np.zeros(60)
for i in range (1000):
    for j in range (4):
        print (arr_iris[i][j])
        \n\

for j in range (60):
    for i in range(100):
        sum_error[j] += arr_error[i]
    #print(sum_error[j]/100)


# In[243]:


#i=0
#x = SGD()
#x.save_data()
#x.h_function(i)
#x.prediction(i)
#x.sigmoid(i)
#x.error(i)
#print (arr_class[i])
#print (arr_sigmoid[i])
#print (arr_error[i])
#x.deltatetha(i)
#for j in range (5):
#    print (deltatetha[i][j])
#x.tetha_baru(i)
#for j in range (5):
#    print (tetha[i+1][j])

