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
            
    #yHat = fns[m-1](AH)  
    return xList, zpList, fns[m-1](AH)  



def testAcc(testData, hList, fns):
    m = len(hList)
    A = testData.X
    for r in range(m):
        if r > 0:
            A = augment(zAH, 1)
        
        AH = A * hList[r]
        zAH = fns[r](AH)
            
    
    try:
        len(testData.labels)
        predictions = [np.argmax(yHat[i,:]) for i in range(nTest)]     
        boolean = [a==b for a, b in zip( testData.labels, predictions)]
        return np.mean(boolean)     
    except(AttributeError):
        return rSqr(testData.Y, testData.Y - zAH)        
        



def getCVsample(D, sampleID, k):
    cvData = namedtuple('data','X Y')
    cvPartition = namedtuple('data', 'R E')
    n = len(sampleID)
    sIndex = [i for i in range(n) if sampleID[i] == k]
    rIndex = [i for i in range(n) if i not in sIndex]
    split = cvPartition(cvData(D.X[rIndex,:], D.Y[rIndex,:]), cvData(D.X[sIndex,:], D.Y[sIndex,:]) )
    return split    


def gradientComputer(hList, gList, xList, zpList, dEdyhat, L1, L2):    
    #L1 =  L2   = 0
    penalty = 0
    m = len(hList)
    s = dim(hList[m - 1])[1]
    for r in range(m-1, -1, -1):
        
        shape = dim(hList[r])
        if r > 0:
            A = augment(xList[r], 1)
        else:
            A = xList[r]
        zeros = np.matrix(np.zeros( dim(zpList[r]) ))   
        
        #E = np.matrix(np.zeros( shape))       
        for i in range(shape[0]):
            for j in range(shape[1]):
                #dyhatdh = np.matrix(np.zeros(dim(zpList[r])))   
                dyhatdh = zeros.copy()   
                dyhatdh[:,j] = np.multiply( A[:,i], zpList[r][:,j])
                #E[i,j] = 1
            
                #dyhatdh = np.multiply( A*E, zpList[r])
                
                for k in range(r+1,m,1):
                    
                    dyhatdh = np.multiply(dyhatdh * hList[k][1:,:], zpList[k])
                dPenalty = 2* L2 * hList[r][i,j] + L1 * np.sign(hList[r][i,j]) 
                
                gList[r][i,j] =  np.sum([dyhatdh[:,i].T*dEdyhat[:,i] for i in range(s)]) + dPenalty 
                
                penalty += L2 * hList[r][i,j]**2 + L1 * abs(hList[r][i,j]) 
                #E[i,j] = 0


    return gList, penalty 

    
def gradComputerExample(hList, gList, xList, zpList, dEdyhat):
    m = len(hList)
    s = dim(hList[m - 1])[1]
    for r in range(m-1, -1, -1):

        #gList[r] *= 0
        shape = dim(hList[r])
        if r > 0:
            A = augment(xList[r], 1)
        else:
            A = xList[r]
        E = np.matrix(np.zeros( shape))                
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                E[i,j] = 1
            
                dyhatdh = np.multiply( A*E, zpList[r])
                
                for k in range(r+1,m,1):
                    dyhatdh = np.multiply(dyhatdh * hList[k][1:,:], zpList[k])
               
                if r == m -1:        
                    ''' This condition is not necessary since E zeros out the column anyway '''
                    gList[r][i,j] = dyhatdh[:,j].T*dEdyhat[:,j]
                else:
                    gList[r][i,j] = np.sum([dyhatdh[:,k].T*dEdyhat[:,k] for k in range(s)])
                E[i,j] = 0
    return gList

def gradComputerOne(gList, xList, zpList, dEdyhat):
    r = 0
    shape = dim(gList[r])
    A = xList[r]             
    
    for j in range(shape[1]):
        for i in range(shape[0]):
            dyhatdh = np.multiply( A[:,i], zpList[r][:,j])
            gList[r][i,j] = dyhatdh.T*dEdyhat[:,j]
            
    return gList
    

def initialize(g, X, fns, dfns):
    initialVars = namedtuple('variables','yHat xList hList gList zpList')    
    ''' Includes bias units '''
    a = .1
    xList = []     
    hList = []
    gList = []
    zpList = []
    Xr = X.copy()

    ''' is the number of mappings between layers '''
    m = len(g) - 1  
    for r in range(m):
        xList.extend([Xr])
        
        if r > 0:
            shape = g[r] + 1, g[r+1]
            A = augment(Xr,0)
        else:
            shape = g[r], g[r+1]
            A = Xr
        H = np.matrix(np.random.uniform(-a, a, shape ))
        hList.extend([H])
        
        gList.extend([a * np.matrix(np.ones(shape))])
        
        AH = A * H
        zpList.extend([ dfns[r](AH) ])
        ''' Forward projection: '''
        Xr = fns[r](AH)    
    
    initialList = initialVars(Xr, xList, hList, gList, zpList)    
    
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
    
def breastCancerData(path):
    dataSet = namedtuple('data','X Y meanY stdY labels')
    f = open(path,'r')
    data = f.read()
    f.close()
    records = data.split('\n')

    n = len(records)-1
    p = len(records[0].split(','))-1
    s = 2 # number of classes
    Y = np.matrix(np.zeros(shape = (n, s)))
    X = np.matrix(np.ones(shape = (n, p)))
    labels = [0]*n
    for i, line in enumerate(records):
        record = line.split(',')
        
        try:
            labels[i] = int(record[p+1]=='4')
            Y[i,labels[i]] = 1
            X[i,1:] =  [int(x)/10 for x  in record[1:p+1]]

        except(ValueError,IndexError):
            pass    
    s = np.std(Y, axis=0)            
    m = np.mean(Y, axis = 0)    
    data = dataSet(X, Y, m, s, [np.argmax(Y[i,:]) for i in range(n)]) 

    return data


def BostonHousing(path):
    #https://archive.ics.uci.edu/ml/datasets/housing
    dataSet = namedtuple('data','X Y meanY stdY labels')
    p = 13 + 1
    f = open(path,'r')
    D = f.read()
    records = D.split('\n')
    n = len(records) - 1
    Y = np.matrix(np.zeros(shape = (n,1)))
    X = np.matrix(np.ones(shape = (n,p)))
    
    endCol = [8,15,23,26,34,43,49,57,61,68,75,82,89,96]
    startCol = [0] + [col+1 for col in endCol[0:13]]
    
    for i, record in enumerate(records):
        
    
        try:
            for j, pair in enumerate(zip(startCol, endCol)):
                
                string = record[pair[0]:pair[1]]
                try:
                    X[i,j+1] = float(string)
                    #print(i,X[i,j+1] )
                except(IndexError):
                    Y[i] = float(string)
        except(ValueError):
            pass
            
    s = np.std(Y, axis=0)            
    m = np.mean(Y, axis = 0)    
    Y = (Y - m)/s
    
    X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0)) / np.std(X[:,1:], axis=0) 
     
    data =  dataSet(X, Y, m, s, None)
    return data

def KingCounty(path):
    ''' from: https://www.kaggle.com/harlfoxem/housesalesprediction '''
    ''' id,date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,
    view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,
    lat,long,sqft_living15,sqft_lot15'''
    dataSet = namedtuple('data','X Y meanY stdY labels')
    f = open(path,'r')
    D = f.read()
    records = D.split('\n')
    n = len(records)
    
    yCol = 2
    colNames = records[0].split(',')
    xCols = [3,4,5,6,7,8,9,10,11,12,13,14,15,19,20]
    zipcodes = [98144]
    
    for i in xCols:
        print(colNames[i])
        
    for i in range(1,len(records)-1):
        data = records[i].split(',')
        zipcode = int(data[16].strip('"'))
        indicator = -1
        notFound = True
        for j,code in enumerate(zipcodes):
            
            if zipcode==code:
                indicator = j
                notFound = False
        if notFound:
            zipcodes.append(zipcode)
            indicator = j+1
            
    n = len(records) - 2  
    p = len(xCols) + len(zipcodes) + 1
    Y = np.matrix(np.zeros(shape = (n,1)))
    X = np.matrix(np.zeros(shape = (n,p)))
    for i in range(1,len(records)-1):
        
        data = records[i].replace('"','').split(',')
        Y[i-1] = float(data[yCol])
        zipcode = int(data[16])        
        X[i-1,0] = 1.0
        for j,index in enumerate(xCols):
            X[i-1,j+1] = float(data[index])
        for j,code in enumerate(zipcodes):
            if zipcode == code:             
                X[i-1,15+j+1] = 1
           
    s = np.std(Y)    
    m = np.mean(Y)    
    Y = (Y - m)/s    
    X[:,1:16] = (X[:,1:16] - np.mean(X[:,1:16], axis=0)) / np.std(X[:,1:16], axis=0) 
    
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


    