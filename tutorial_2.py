# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 08:54:41 2018

@author: brian
"""
import random
import numpy as np
class ActivationFunction(object):
    def __init__(self, function, derivative):    
        self.function = function
        self.derivative = derivative
    
    def evaluate(self, X):
        shape = np.shape(X)
        A = np.matrix(np.zeros(shape))
        for i in range(shape[0]):    
            for j in range(shape[1]):  
                A[i,j] = self.function(X[i,j])
        return A
            
    def differentiate(self, X):
        shape = np.shape(X)
        A = np.matrix(np.zeros(shape))
        for i in range(shape[0]):    
            for j in range(shape[1]):  
                A[i,j] = self.derivative(X[i,j])
        return A

def identity(x):
    return x
def unit(x):
    return 1
def dim(X):
    return np.shape(X)

def rlu(x):
    return max([0,x])    
def rluP(x):
    return int(x > 0)    

''' Activation function test: '''    

p = q = 3
X = np.matrix(np.zeros( (p, q)) )
for i in range(p):
    for j in range(q):
        X[i,j] = random.choice(np.arange(-10, 10, 1)) 

z0 = ActivationFunction(rlu, rluP)   
z1 = ActivationFunction(identity, unit)   

fns = [z0.evaluate, z1.evaluate]
dfns = [z0.differentiate, z1.differentiate]

path = '/home/brian/NeuralNets/Data/parkinsons_updrs.csv'
from functions import parkinsonsData 
D = parkinsonsData(path)
n, p = dim(D.X)
n, s = dim(D.Y)
print(n, p, s)

g = [p,  s]
K = 10

from functions import getCVsample
from functions import rSqr
from functions import initialize

def testAcc(testData, hList, fns):
    m = len(hList)
    A = testData.X
    for r in range(m):
        if r > 0:
            A = augment(zAH, 1)
            
        AH = A * hList[r]
        zAH = fns[r](AH)
    return rSqr(testData.Y, testData.Y - zAH)   

def gradComputerOne(gList, xList, zpList, dEdyhat):
    r = 0
    shape = dim(gList[r])
    A = xList[r]
    for j in range(shape[1]):
        for i in range(shape[0]):
            dyhatdh = np.multiply(A[:,i], zpList[r][:,j])
            gList[r][i,j] = dyhatdh.T*dEdyhat[:,j]    
    return gList

def fProp(xList, hList, fns, dfns, zpList):
    m = len(xList)
    A = xList[0]
    for r in range(m):
        if r > 0:
            A = augment(xList[r], 1)
        AH = A * hList[r]
        if r < m -1:
            xList[r+1] = fns[r](AH)
        zpList[r] = dfns[r](AH)
    yHat = fns[m-1](AH)
    return xList, zpList, yHat

seed = 0
random.seed(seed)
np.random.seed(seed)
    
sampleID = [random.choice(range(K)) for i in range(n)]
for k in range(K):

    try:
        del progress
    except(NameError):
        pass
    
    F = getCVsample(D, sampleID, k)    
    X = F.R.X
    beta = np.linalg.solve(X.T*X, X.T*F.R.Y)   
    #print(beta)
    E = F.E.Y - F.E.X*beta
    rsq = rSqr(F.E.Y, E) 
    print('N train = ',dim(F.R.Y)[0], '\tn test = ',dim(F.E.Y)[0] ,
      '\tLinear regression adj R2 (test) = ',rsq)
    gamma = .00001 
    yHat, xList, hList, gList, zpList = initialize(g, F.R.X, fns, dfns)
        
    it = 0    
    while it < 1000: 

        xList, zpList, yHat = fProp(xList, hList, fns, dfns, zpList)
        
        dEdyhat = -2*(F.R.Y - yHat)
        
        gList = gradComputerOne(gList, xList, zpList, dEdyhat)
        hList[r] -= np.multiply(gamma, gList[r])
        
                                    
        obsAcc = testAcc(F.E, hList, fns)
        objFunction = sum([0.25* np.mean([x**2 for x in dEdyhat[:,i]]) 
            for i in range(s)])
        obsAcc.append(objFunction)
        
        try:
            progress = [.9*progress[i] + .1*obsAcc[i] for i in range(len(progress))]
        except(NameError):
            progress = obsAcc
                    
        string = '\r'+str(k) + '/' + str(K) + ' \t' + str(it)
        for i in range(s):
            string += '\t r-sqr = '+ str(round(progress[i],3)) 
        string +='\t obj = '+ str(round(progress[len(progress)-1],5))
        print(string, end="")
        
        it += 1