"""
Assisgnment 12: Building a Neural Network and a Support Vector Machine models 
without ML libraries to classsify digit "1" from all other digits within MNIST data set

PART 2: Support Vector Machine
"""

#Importing libraries inclsuing solver to solve the optimal separating hyperplane
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

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


#Building an 8th degree polynomial kernel
C = 25
def kernel(x,y):
    kern = ((x @ y.T) + 1) ** 8
    return kern
Xtrain_trans = kernel(Xtrain,Xtrain) 

#Building the matrices to reach the format needed for the solver
Ymult= (Ytrain.reshape(Ytrain.shape[0],1) @ Ytrain.reshape(Ytrain.shape[0],1).T)
Qd = Ymult *  Xtrain_trans
Ad = np.vstack((-Ytrain.T,Ytrain.T,-np.eye(Ytrain.shape[0],Ytrain.shape[0]),np.eye(Ytrain.shape[0],Ytrain.shape[0])))
Pd = -np.ones((Ytrain.shape[0],1))
cmat = np.vstack((np.zeros((Ytrain.shape[0]+2,1)),np.ones((Ytrain.shape[0],1))*C))
Q = matrix(Qd, tc='d')
p = matrix(Pd, tc='d')
G = matrix(Ad, tc='d')
h = matrix(cmat, tc='d')
solvers.options['abstol'] = 1e-11 # <<<
solvers.options['reltol'] = 1e-11 # <<<
solvers.options['feastol'] = 1e-12 # <<<

sol = solvers.qp(P=Q,q=p,G=G,h=h)
alpha =np.array( sol['x'])

sv_ind = np.where(np.any(alpha>0.000001, axis=1))
alpha_s = alpha[sv_ind]
y_s = Ytrain[sv_ind]
x_s = Xtrain[sv_ind]
mask = alpha>0.00001
b=0

x_s_kernel = kernel(x_s,x_s[0]) 
b = (y_s[0] -  np.sum(y_s.reshape(y_s.shape[0],1)*alpha_s*x_s_kernel.reshape(y_s.shape[0],1)))
 
 
#Producing the charts for the data points along with the classification
#Building the classification regions
xDim= np.arange(-1.1, 1.1, 0.01)
yDim= np.arange(-1.1, 1.1, 0.01)
X,Y = np.meshgrid(xDim, yDim)
grid = np.column_stack((X.flatten(),Y.flatten()))
color= []
xgrid = []
ygrid = []
hgrid_list =[]
for i in range(grid.shape[0]):
    g_x = np.sum(y_s.reshape(y_s.shape[0],1)*alpha_s*kernel(x_s,grid[i]).reshape(y_s.shape[0],1))+b
    hgrid = np.sign(g_x)
    if hgrid >0:
        color.append('#d8d8ff')
    else:
        color.append('#ffd8d8')
    xgrid.append(grid[i,0])
    ygrid.append(grid[i,1])
    hgrid_list.append(hgrid)
plt.scatter(xgrid,ygrid, c=color, marker='.', lw=0)

#Drawing the classified data points colored within their classified regions
for i in range(Xtrain.shape[0]):
    if   Ytrain[i] == 1:
        plt.scatter(Xtrain[i,0], Xtrain[i,1], marker="o", facecolors='none', color='blue',s=20,label='1')
    elif Ytrain[i] == -1:
        plt.scatter(Xtrain[i,0], Xtrain[i,1], marker="x", color='red',s=20, label='Others')

#Calculating the error within the test data set 
E_t = 0
for i in range(Ytest.shape[0]):
    g_test = np.sum(y_s.reshape(y_s.shape[0],1)*alpha_s*kernel(x_s,Xtest[i]).reshape(y_s.shape[0],1))+b
    ht = np.sign(g_test)
    if ht !=  Ytest[i]:
        E_t += 1
Etest = (1/(Ytest.shape[0]))*E_t 
    