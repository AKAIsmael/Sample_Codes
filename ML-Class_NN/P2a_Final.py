"""
Assisgnment 12: Building a Neural Network and a Support Vector Machine models 
without ML libraries to classsify digit "1" from all other digits within MNIST data set

PART 1: Neural Network 
"""

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime

#Reading input files
train_data = pd.read_csv("ZipDigits.train.txt", sep=" ", header=None)
train_data.rename(columns={0: "Number"}, inplace = True)
test = pd.read_csv("ZipDigits.test.txt", sep=" ", header=None)
test.rename(columns={0: "Number"}, inplace = True)

"""
Aggregating train and test data for transformations to build the features used for 
classification: symmetry. Afterwards the values are normalized 
"""
data=[train_data,test]
dataset = pd.concat(data)
dataset=dataset.reset_index()
features = np.zeros((dataset.shape[0],3))
arr=np.zeros((258))
y=[]
for ind in dataset.index:
    features[ind,0]=dataset['Number'][ind]
    if features[ind,0] ==1:
        y.append(1)
    else:
        y.append(-1)
    arr=dataset.iloc[ind].values
    arr2=arr[1:257].reshape(16,16)
    flip_arr = np.flip(arr2,axis=0)
    sym= -np.average(np.abs(arr2-flip_arr))
    features[ind,1]=sym
    features[ind,2]=np.average(arr[1:257])
min_ft = np.amin(features[:,1:],axis=0)
max_ft =np.amax(features[:,1:],axis=0)
features[:,1]=2*(features[:,1]-min_ft[0]) / (max_ft[0]-min_ft[0]) -1
features[:,2]=2*(features[:,2]-min_ft[1]) / (max_ft[1]-min_ft[1]) -1

#Dividing dataset to test and train data sets
X=np.column_stack((features[:,2],features[:,1]))
Y=np.array(y)
np.random.seed(390)
tr_idx = np.random.choice(X.shape[0], 300 ,replace=False)
Xtrain = X[tr_idx]
Ytrain= Y[tr_idx]
Xtest = X[[i for i in range(X.shape[0]) if i not in tr_idx ]]
Ytest = Y[[i for i in range(X.shape[0]) if i not in tr_idx ]]

# Inititating Weights, using 10 hidden units
m=10
w1 = np.random.randn(3,m)*0.1
w2 = np.random.randn(m+1,1)*0.1
x0 = np.column_stack((np.ones(300),Xtrain)).T

#A forward propagation with tanh activation function
def forward_prop (m,x0,w1,w2):
    s1 = np.dot(w1.T,x0)
    x1_temp = np.tanh(s1)
    x1 = np.vstack((np.ones(x0.shape[1]),x1_temp))
    s2 = np.dot(w2.T,x1)
    x2 = s2                            

    return x1, x2
x1,x2 = forward_prop(m, x0, w1, w2)

#A backward propagation (calculating derivatives)
def back_prop (m,x0,x1,x2,w2,y):                  
    delta_2 = 2 *(x2-y) 
    dw2 = np.dot(x1,delta_2.T)
    theta = (1-(x1[1:,:])**2)
    delta_1 = np.multiply(theta, np.dot(w2,delta_2)[1:])
    dw1 = np.dot(x0,delta_1.T)
    return dw1/(4*x2.shape[1]), dw2/(4*x2.shape[1]), delta_1, delta_2

#Calculating In-sample error
def Ein_compt(x2,y):
    h = np.sign(x2)
    return (1/(4*x2.shape[1])) * np.sum((x2-y)**2)

#Updating weights
def update_wts (w1,w2,lr,dw1,dw2):
    wa = w1.copy()
    wb = w2.copy()
    wa -= lr * dw1
    wb -= lr * dw2
    return wa,wb

#Defining maximum iterations, parameters and initating in-sample error values array
iter = 2000000
alpha = 1.1
beta = 0.7
lr = 0.1
Ein=[]
x1,x2 = forward_prop(m, x0, w1, w2) 
dw1,dw2,delta_1, delta_2 = back_prop (m,x0,x1,x2,w2,Ytrain)
Ein_old = Ein_compt(x2,Ytrain)
Ein.append(Ein_old)
count = 0

#Running for max iterations while adjusting step size for gardient descent
start = datetime.datetime.now() 
while iter > 0:
    w1n,w2n =  update_wts (w1,w2,lr,dw1,dw2)
    z1,z2 = forward_prop(m, x0, w1n, w2n) 
    dw1n,dw2n,delta_1, delta_2 = back_prop(m,x0,z1,z2,w2n,Ytrain)
    Ein_new = Ein_compt(z2,Ytrain)
    if Ein_new < Ein_old:
        w1, w2 = w1n, w2n
        dw1, dw2 = dw1n, dw2n
        lr *= alpha
        Ein.append(Ein_new)
        Ein_old = Ein_new
        print(iter, "cond_1")
    else:
        lr *= beta    
        print(iter, "cond_2")
    iter -= 1
    count += 1
time= (datetime.datetime.now()-start)   

#Ploting the in-sample error
plt.semilogy(Ein)
plt.show()

#Calculating the error within the test data set
xtest0 = np.column_stack((np.ones(8998),Xtest)).T
g1,g2 = forward_prop(m, xtest0, w1, w2)
def Etest_compt(x2,y):
    error = 0
    h = np.sign(x2)
    for i in range(x2.shape[1]):
        if h[0,i] !=  y[i]:
            error += 1
    return (1/(x2.shape[1])) * error
Etest = Etest_compt(g2,Ytest)

#Producing the charts for the data points along with the classification
#Building the classification regions
xDim= np.arange(-1.1, 1.1, 0.01)
yDim= np.arange(-1.1, 1.1, 0.01)
X,Y = np.meshgrid(xDim, yDim)
grid = np.column_stack((X.flatten(),Y.flatten()))
grid0 = np.column_stack((np.ones(grid.shape[0]),grid)).T
grid1,grid2 = forward_prop(m, grid0, w1, w2)
hgrid = np.sign(grid2)
color= []
xgrid = []
ygrid = []
for i in range(grid2.shape[1]):
    if hgrid[0,i] >0:
        color.append('#d8d8ff')
    else:
        color.append('#ffd8d8')
    xgrid.append(grid[i,0])
    ygrid.append(grid[i,1])
plt.scatter(xgrid,ygrid, c=color, marker='.', lw=0)

#Drawing the classified data points colored within their classified regions
for i in range(Xtrain.shape[0]):
    if   Ytrain[i] == 1:
        plt.scatter(Xtrain[i,0], Xtrain[i,1], marker="o", facecolors='none', color='blue',s=20,label='1')
    elif Ytrain[i] == -1:
        plt.scatter(Xtrain[i,0], Xtrain[i,1], marker="x", color='red',s=20, label='Others')














 

