# Important libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
import math
class Data:
  
  def __init__(self,fileName):
    self.loadData(fileName)
    color = self.y@np.arange(1, self.y.shape[1]+1, 1, dtype=int).T
    self.color = color

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

  def SplitData(self,testRatio):
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
    # Plots the variables two by two in matrix of correlation
    nF = self.X.shape[1]
    figure, axes = plt.subplots(nrows=nF, ncols=nF)
    plt.gcf().set_size_inches(10, 8)
    for i in range(nF):
      for j in range(nF):
        if i == j:
          axes[i,j].hist(self.X[:,j])
        else:
          axes[i,j].scatter(self.X[:,j],self.X[:,i],c=self.color,cmap=cm.brg)
    plt.show()  

  def plotData(self,i,j):
    # Plots Xi vs Xj
    plt.scatter(self.X[:,i],self.X[:,j],c=self.color,cmap=cm.brg)

def testAccuracy(Z,Y):
  cont = 0
  for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
      if Z[i,j] == np.amax(Z[i,:]):
        p = j
      if Y[i,j] == np.amax(Y[i,:]):
        t = j
    if p == t:
      cont += 1 
  TA = cont/Y.shape[0]
  return TA