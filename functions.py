# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 10:22:03 2018

@author: brian
"""
import numpy as np
from collections import namedtuple
initialParameters = namedtuple('initialization','xList hList gradients zpList stepSizes norms vList iList')
''' ******************** Activation functions ****************************'''        

def elu(x):
    ''' https://en.wikipedia.org/wiki/Rectifier_(neural_networks)'''
    I = int(x > 0)
    return I*x + 1E-5*(1 - I)*(np.exp(x) - 1)
def eluP(x):
    I = int(x > 0)
    return I + (1 - I)* 1E-5* np.exp(x)

def identity(x):
    return x
def unit(x):
    return 1

def logistic(x):
    return softPlusP(x)
    
def logisticP(x):
    return logistic(x)*(1- logistic(x))    
    
def rlu(x):
    return max([0,x])    
def rluP(x):
    return int(x > 0) 

def softPlus(x):
    return np.log(1 + np.exp(x) )
def softPlusP(x):
    return 1/(1 + np.exp(-x) )

def tanhP(x):
    return 1 - np.tanh(x)**2    


class ActivationFunction(object):
    def __init__(self, function, derivative):    
        self.function = function
        self.derivative = derivative
            
    def differentiate(self, X):
        shape = np.shape(X)
        A = np.matrix(np.zeros(shape))
        for i in range(shape[0]):    
            for j in range(shape[1]):  
                A[i,j] = self.derivative(X[i,j])
        return A
    
    def evaluate(self, X):
        shape = np.shape(X)
        A = np.matrix(np.zeros(shape))
        for i in range(shape[0]):    
            for j in range(shape[1]):  
                A[i,j] = self.function(X[i,j])
        return A
    
    
def batchSample(batSize, iteration, indx, xList, trainSet):
    if len(indx) < batSize:
        sample = indx
    else:
        sample = [indx[i % len(indx)] for i in range(batSize*iteration, batSize*(iteration + 1))]
        
    try:
        labelsSample = [trainSet.labels[i] for i in sample]
    except(TypeError, AttributeError):
        labelsSample = 0
    try:
        xList[len(xList)-1] = trainSet.X[sample,:]
        return xList, trainSet.Y[sample,:], labelsSample
    except(TypeError, NameError):
        return trainSet.X[sample,:], trainSet.Y[sample,:], labelsSample

def augment(X, value):
    n, _ = np.shape(X)
    return np.hstack((value*np.ones(shape= (n,1)), X ))    

def dEdyhatSqr(Y, yHat):
    return -2 * (Y - yHat)
        
def dim(X):
    return np.shape(X)


    
def forwardPropagation(xList, hList, fns, dfns, zpList):
    nH = len(xList) - 1
    for r in range(nH, -1, -1):
        #print(r)
        if r < nH:
            A = augment(xList[r], 1)
        else:
            A = xList[r]
        
        AH = A * hList[r]
        
        zpList[r] = dfns[r](AH)
        if r > 0:
            xList[r-1] = fns[r](AH)
    yHat = fns[0](AH)  
    return xList, zpList, yHat

def testAcc(testData, hList, fns):
    
    nH = len(hList) - 1
    for r in range(nH, -1, -1):
        if r < nH:
            A = augment(X, 1)
        else:
            A = testData.X
        AH = A * hList[r]
        if r > 0:
            X = fns[r](AH)
    yHat = fns[0](AH)  
    
    try:
        len(testData.labels)
        predictions = [np.argmax(yHat[i,:]) for i in range(nTest)]     
        boolean = [a==b for a, b in zip( testData.labels, predictions)]
        return np.mean(boolean)     
    except(AttributeError):
        return rSqr(testData.Y, testData.Y - yHat)
        
def getCVsample(D, sampleID, k, cvData):
    cvData = namedtuple('data','Xr Yr Xe Ye')
    n = len(sampleID)
    sIndex = [i for i in range(n) if sampleID[i] == k]
    rIndex = [i for i in range(n) if i not in sIndex]
    return cvData(D.X[rIndex,:], D.Y[rIndex,:], D.X[sIndex,:], D.Y[sIndex,:])


    
def gradComputerExample(hList, gList, xList, zpList, dEdyhat):
    L = len(hList)
    s = dim(hList[0])[1]
    for r in range(L):

        gList[r] *= 0
        shape = dim(hList[r])
        if r < L - 1:
            A = augment(xList[r], 1)
        else:
            A = xList[r]
        E = np.matrix(np.zeros( shape))                
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                E[i,j] = 1
            
                dyhatdh = np.multiply( A*E, zpList[r])
                for k in range(r-1,-1,-1):
                    #dyhatdh = np.multiply(dyhatdh * hList[k][1:,:], zpList[k])
                    dyhatdh = np.multiply(augment(dyhatdh, 0) * hList[k], zpList[k])
                        
                gList[r][i,j] = np.sum([dyhatdh[:,i].T*dEdyhat[:,i] for i in range(s)])
                E[i,j] = 0
    return gList

def gradComputerOne(hList, gList, xList, zpList, dEdyhat):
    L = len(hList)
    r = 0
    s = dim(hList[0])[1]

    gList[r] *= 0
    shape = dim(hList[r])
    
    A = xList[r]
    E = np.matrix(np.zeros( shape))                
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            E[i,j] = 1
        
            dyhatdh = np.multiply( A*E, zpList[r])
            gList[r][i,j] = np.sum([dyhatdh[:,i].T*dEdyhat[:,i] for i in range(s)])
            E[i,j] = 0
    return gList
    
def initialize(g, X, fns, dfns):
    initialVars = namedtuple('variables','yHat xList hList gList zpList')    
    ''' Includes bias units '''
    a = .1
    xList = []
    hList = []
    gList = []
    zpList = []
    Xi = X

    ''' is the number of mappings between layers '''
    m = len(g) - 1  
    for r in range(m):
        
        if r > 0:
            shape = g[r] + 1, g[r+1]
            A = augment(Xi,0)
        else:
            shape = g[r], g[r+1]
            A = Xi
        xList.extend([Xi])
        
        H = np.matrix(np.random.uniform(-a, a, shape ))
        hList.extend([H])
        
        G = a * np.matrix(np.ones(shape))
        gList.extend([G])
        
        AH = A * H
        Xi = fns[m-r-1](AH)
        
        zpList.extend([ dfns[r](AH) ])
        
    xList = xList[::-1]    
    hList = hList[::-1]    
    gList = gList[::-1]    
    zpList = zpList[::-1]   
     
    initialList = initialVars(Xi, xList, hList, gList, zpList)    
    
    return initialList 

    
def parkinsonsData(path):
    
    dataSet = namedtuple('data','X Y meanY stdY labels')
    records =  open(path,'r').read().split('\n')
    variables = records[0].split(',')
    
    iX = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    iY = [4, 5]
    
    print('Predictor variables:')    
    for i in range(len(iX)) : 
        print(iX[i], variables[iX[i]])
    print('Target variables:')    
    for i in range(len(iY)) : 
        print(iY[i], variables[iY[i]])
        
    n = len(records)-1
    p = len(iX) + 1
    try:
        s = len(iY)
    except(TypeError):
        s = 1
    
    Y = np.matrix(np.zeros(shape = (n, s)))
    X = np.matrix(np.ones(shape = (n, p )))
    for i, j in enumerate(np.arange(1,n+1,1)):
        lst = records[j].split(',')
        for k in range(s):
            Y[i,k] = float(lst[iY[k]])
        for k in range(p-1):
            X[i,k+1] = lst[iX[k]]    
    
    s = np.std(Y, axis=0)            
    m = np.mean(Y, axis = 0)    
    Y = (Y - m)/s
    
    X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0)) / np.std(X[:,1:], axis=0)            

    data = dataSet(X, Y, m, s, None)
    return data

def rmseFn(Y, yHat, n): 
    s = np.shape(yHat)[1]
    return float(sum([sum(np.multiply(Y[:,i] - yHat[:,i],Y[:,i] - yHat[:,i])) for i in range(s)])/n)

def RMSProp2(prevGrad, currGrad, a):
    rho = 0.9
    sqr = rho*np.power(prevGrad, 2)  + (1 - rho)*np.power(currGrad, 2)
    return a/np.sqrt(sqr + 1e-6)
        
def rSqr(Y, E):
    varY = np.var(Y, axis = 0)
    varE = np.var(E, axis = 0)
    lst = np.matrix.tolist(1 - varE/varY)[0]

    return [round(float(r),4) for r in  lst]


    

def getCVsample(D, sampleID, fold):
    cvData = namedtuple('data','X Y')
    cvPartition = namedtuple('data', 'R E')
    n = len(sampleID)
    sIndex = [i for i in range(n) if sampleID[i] == fold]
    rIndex = [i for i in range(n) if i not in sIndex]
    split = cvPartition(cvData(D.X[rIndex,:], D.Y[rIndex,:]), cvData(D.X[sIndex,:], D.Y[sIndex,:]) )
    return split    