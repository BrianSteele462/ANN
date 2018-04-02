# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 10:15:20 2018

@author: brian
"""

import numpy as np
import sys
import random
from collections import namedtuple

#import functions as fn
from functions import *
sys.path.append('/home/brian/NeuralNets/PythonScripts')

def identity(x):
    return x
def unit(x):
    return 1

def rlu(x):
    return max([0,x])    
def rluP(x):
    return int(x > 0) 

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
    
z0 = ActivationFunction(rlu, rluP)   
z1 = ActivationFunction(identity, unit)   

fns = [z0.evaluate, z1.evaluate]
dfns = [z0.differentiate, z1.differentiate]

def dEdyhatSqr(Y, yHat):
    return -2 * (Y - yHat)

def gradComputerOne(gList, xList, zpList, dEdyhat):
    r = 0
    shape = dim(gList[r])
    A = xList[r]             
    
    for j in range(shape[1]):
        for i in range(shape[0]):
            dyhatdh = np.multiply( A[:,i], zpList[r][:,j])
            gList[r][i,j] = dyhatdh.T*dEdyhat[:,j]
            
    return gList
seed = 0
random.seed(seed)
np.random.seed(seed)
'''  START computing ... '''    

path = '/home/brian/NeuralNets/Data/bostonHousingData.txt'
D = BostonHousing(path)
        


path = '/home/brian/NeuralNets/Data/parkinsons_updrs.csv'

D = parkinsonsData(path)
path =  '/home/brian/NeuralNets/Data/kc_house_data.csv'
D = KingCounty(path)

n, p = dim(D.X)
n, s = dim(D.Y)
print(n,p, s)

q = p
g = [p,  s]
K = 10

a = .001
cvDict = dict.fromkeys(range(K))
sampleID = [random.choice(range(K)) for i in range(n)]
m = len(g) - 1  # number of maps between layers 
for k in range(K):
    try:
        del progress
    except(NameError):
        pass
    
    F = getCVsample(D, sampleID, k)    
    
    X = F.R.X
    Y = F.R.Y
    
    beta = np.linalg.solve(X.T*X, X.T*Y) 
    E = F.E.Y - F.E.X*beta
    rsq = rSqr(F.E.Y, E)    
    
    print('N train = ',dim(F.R.Y)[0], '\tn test = ',dim(F.E.Y)[0] ,
             '\tLinear regression adj R2 (test) = ',rsq)
    
    initialList = initialize(g, F.R.X, fns, dfns)
    yHat, xList, hList, gList, zpList = initialList  
    
    #  Begin iterations 
    it = 0    
    epoch = 0
    while it < 2000: 
        
        if it % 101 == 0 and a > 1E-6:
            a = a/2
        # Forward Propagation 
        xList, zpList, yHat = fProp(xList, hList, fns, dfns, zpList)
        dEdyhat = -2 * (F.R.Y - yHat) 
        
        # Compute gList 
        prevGradList = gList.copy()   
        if m > 1:
            gList = gradComputerExample(hList, gList, xList, zpList, dEdyhat)    
            
        else:
            gList = gradComputerOne(gList, xList, zpList, dEdyhat)    
                    
                

        ''' Compute the step sizes '''
        a = .001
        for r in range(m):
            
            stepSize = RMSProp2(prevGradList[r], gList[r], a)
            #stepSize, _ = fn.RMSProp(prevGradList[r], gList[r], a)
            
            hList[r] -= np.multiply(stepSize, gList[r])
            #hList[r] -= np.multiply(.00001, gList[r])

        #0/10    999      r-sqr = 0.159   r-sqr = 0.165 

        # Evaluate progress 
        
        obsAcc = testAcc(F.E, hList, fns)
        objFunction = sum([0.25* np.mean([x**2 for x in dEdyhat[:,i]]) for i in range(s)])
        obsAcc.append(objFunction)
        
        try:
            progress = [.9*progress[i] + .1*obsAcc[i] for i in range(len(progress))]
        except(NameError):
            progress = obsAcc
            
        string = '\r'+str(k) + '/'+str(K)+' \t'+str(it)
        for i in range(s):
            string += '\t r-sqr = '+ str(round(progress[i],3)) 
        string +='\t obj = '+ str(round(progress[len(progress)-1],5))
        print(string,end="")
    
        it += 1
    
    #print('Completed ', k+1, 'of ', K)  
    cvDict[k] = [dim(F.E.Y)[0], rsq]   
cvAcc = [sum([cvDict[k][0] * cvDict[k][1][i] for k in range(K)])/n for i in range(2)]    
print('\nCV adjusted R-sqr = '+'  '+'  '.join([str(round(a,3)) for a in cvAcc]))

sys.exit()
np.savetxt('/home/brian/M462/Data/H0.txt' ,hList[0],delimiter =',')
np.savetxt('/home/brian/M462/Data/H1.txt' ,hList[1],delimiter =',')