import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm 
import math

class Data:
  def __init__(self,fileName):
    self.loadData(fileName)
    self.splitData(0)

  def loadData(self,fileName):
    # Uses a csv file to create a numpy array
    with open(fileName, newline='') as csvfile:
      reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      data = list(reader)
    data = np.array(data)
    self.X = np.array(data[1:np.shape(data)[0],0:-1],dtype='float64')  
    ydata = np.array(data[1:np.shape(data)[0],-1],dtype='int32')
    self.y = np.zeros((len(ydata),np.amax(ydata)-np.amin(ydata)+1))
    for i in range(self.y.shape[0]):
      for j in range(self.y.shape[1]):
        if ydata[i] == j+1:
            self.y[i,j] = 1

  def splitData(self,testRatio):
    # Shuffles the data and splits data in train and test 
    M      = self.X.shape[0]
    ntrain = int((1-testRatio)*M) 
    idx    = np.arange(M)
    np.random.shuffle(idx) 
    self.Xtrain = self.X[idx[0:ntrain],:]
    self.Xtest  = self.X[idx[ntrain:M],:]
    self.Ytrain = self.y[idx[0:ntrain],:]
    self.Ytest  = self.y[idx[ntrain:M],:]

  def plotCorrelationMatrix(self):
    # Plots the variables two by two
    nF = self.X.shape[1]
    figure, axes = plt.subplots(nrows=nF, ncols=nF)
    plt.gcf().set_size_inches(10, 8)
    color = self.y@np.arange(1, self.y.shape[1]+1, 1, dtype=int).T
    self.color = color
    for i in range(nF):
      for j in range(nF):
        if i == j:
          axes[i,j].hist(self.X[:,j])
        else:
          axes[i,j].scatter(self.X[:,j],self.X[:,i],c=self.color,cmap=cm.brg)
    plt.show()  

  def plotData(self,i,j):
    # It plots Xi vs Xj
    plt.scatter(self.X[:,i],self.X[:,j],c=self.color,cmap=cm.brg)
  
  def scale(self):
    # Scales the data to a (0,1) interval
    for i in range(self.X.shape[1]):
      self.X[:,i] = (self.X[:,i] - np.amin(self.X[:,i]))/(np.amax(self.X[:,i])-np.amin(self.X[:,i]))

def buildmodel(Xtr,Ytr,d,idx):
  comb = int(math.factorial(d+2)/(math.factorial(d)*math.factorial(2)))
  X    = np.ones((Xtr.shape[0],comb))
  cont = 1
  for i in range(1,d+1):
    for j in range(0,i+1):
      X[:,cont]  = Xtr[:,idx[0]]**(i-j)*Xtr[:,idx[1]]**(j)
      cont += 1
  Y = Ytr
  u = (6/(X.shape[1]+Y.shape[1]))**0.5
  theta = np.random.uniform(-u,u,(X.shape[1]*Y.shape[1],1))
  return theta,X,Y

# This function plots the boundaries for the model
def PlotBoundary(data,theta,d,idx):
    X = data.X
    Y = data.y
    n_points = 200
    x1 = np.linspace(min(X[:,idx[0]]),max(X[:,idx[0]]),n_points).T
    x1 = np.reshape(x1,(n_points,1))
    x2 = np.zeros((n_points,1)) + min(X[:,idx[1]])
    y = np.zeros((n_points,3))
    x2_aux = np.zeros((n_points,1))
    xtest = np.zeros((n_points,theta.shape[0],n_points))
    h = np.zeros((n_points*Y.shape[1],n_points))
    for i in range(n_points):
        x2 = x2 + (max(X[:,1])-min(X[:,1]))/n_points
        x2_aux[i] = x2[0]
        xdata_test = np.concatenate((x1,x2),axis=1)
        _,xtest[:,:,i],_ = buildmodel(xdata_test,Y,d,[0,1]) 
        haux = hypothesisFunction(xtest[:,:,i],theta)
        h[:,i] = np.reshape(haux.T,(n_points*Y.shape[1],))
    data.plotData(idx[0],idx[1])
    x1_draw = np.reshape(x1,(n_points,))
    x2_draw = np.reshape(x2_aux,(n_points,))
    x1_draw,x2_draw = np.meshgrid(x1_draw,x2_draw,indexing='ij')
    cl = ['blue','green','red']
    for j in range(Y.shape[1]):
        h_draw = h[(j*n_points):((j+1)*n_points),:]    
        plt.contour(x1_draw,x2_draw,h_draw,levels = [0],colors = cl[j])

# Given X,Y and theta, this function tests the percentage of success
def testAccuracy(X,Y,theta):
  h = hypothesisFunction(X,theta)
  g = sigmoid(h)
  cont = 0
  for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
      if g[i,j] == np.amax(g[i,:]):
        p = j
      if Y[i,j] == np.amax(Y[i,:]):
        t = j
    if p == t:
      cont += 1 
  TA = cont/Y.shape[0]
  return TA

def hypothesisFunction(X,theta):
  h = X@theta
  return h
def sigmoid(z):
  g = 1/(1+math.e**(-z))
  return g