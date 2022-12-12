#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import numpy as np
from matplotlib import pyplot as plt

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
    self.X = np.array(data[1:np.shape(data)[0],:],dtype='float64')  
    
  def splitData(self,testRatio):
    # Shuffles the data and splits data in train and test 
    M      = self.X.shape[0]
    ntrain = int((1-testRatio)*M) 
    idx    = np.arange(M)
    np.random.shuffle(idx) 
    self.Xtrain = self.X[idx[0:ntrain],:]
    self.Xtest  = self.X[idx[ntrain:M],:]

  def plotData(self):
    # Plots de data in a 2D graph
    nF = self.X.shape[1]
    plt.plot(self.Xtrain[:,0],self.Xtrain[:,1],'b.')
    plt.plot(self.Xtest[:,0],self.Xtest[:,1],'r.')
    if self.Xtest.shape[0] > 0:
      plt.legend(['Train','Test'])

  def scale(self):
    # Scales the data to a (0,1) interval
    for i in range(self.X.shape[1]):
      self.X[:,i] = (self.X[:,i] - np.amin(self.X[:,i]))/(np.amax(self.X[:,i])-np.amin(self.X[:,i]))

