# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:55:48 2018

@author: brian

"""

func_0 = ActivationFunction(fn.identity, fn.unit)   
func_1 = ActivationFunction(fn.softPlus, fn.softPlusP)   
func_2 = ActivationFunction(fn.elu, fn.eluP)   
func_3 = ActivationFunction(fn.np.tanh, fn.tanhP)   
func_4 = ActivationFunction(fn.logistic, fn.logisticP)   
func_5 = ActivationFunction(fn.rlu, fn.rluP)   

import random
import numpy as  np
import sys
print(sys.version_info)
import copy

from collections import namedtuple
from collections import defaultdict

''' Order: output to input '''
sys.path.append("/home/brian/NeuralNets/PythonScripts")
import functionsNN as fn  
from functions import parkinsonsData
from functions import ActivationFunction 
from functions import dim
dataSet = namedtuple('data','X Y meanY stdY labels')
cvData = namedtuple('data','Xr Yr Xe Ye')

path = '/home/brian/NeuralNets/Data/parkinsons_updrs.csv'
D = parkinsonsData(path)

#D =  BostonHousing('/home/brian/NeuralNets/Data/bostonHousingData.txt')
n, p = dim(D.X)
n, s = dim(D.Y)
print(n,p, s)

func_0 = ActivationFunction(fn.identity, fn.unit)   
func_1 = ActivationFunction(fn.softPlus, fn.softPlusP)   
func_2 = ActivationFunction(fn.elu, fn.eluP)   
func_3 = ActivationFunction(fn.np.tanh, fn.tanhP)   
func_4 = ActivationFunction(fn.logistic, fn.logisticP)   
func_5 = ActivationFunction(fn.rlu, fn.rluP)   
fns = [func_0.evaluate,  func_2.evaluate,       func_5.evaluate, func_2.evaluate]
dfns = [func_0.differentiate, func_2.differentiate, func_2.differentiate, func_5.differentiate , func_2.differentiate]

b = .1
q = 30
g = [p, 30,  s] # func_2 0.60644

g = [p, p, s] # 
a = .01
batSize = 100
L1 = L2 = .000
objFn = fn.rmseFn
dEdyhatf = fn.dEdyhatSqr

K = 5
''' No specification below ... '''
L = len(g) - 1
nH = L - 1
print('n hidden hList = ', L)
''' L is the number of hidden hList + 1. (1 counts the input layer X) '''

 
random.seed(0)
np.random.seed(0)

sampleID = [random.choice(range(K)) for i in range(n)]
folds = [i for i in range(K)]


lrDict = {}
cvDict = {}
objList = [0]*20
accList = [0]*20
aConstant = .005  # for boston
aConstant = .001  # for parkinsons
for k in range(K):
    a = aConstant
    #trainSet, testSet = fn.validationSample(D, sampleID, fold, dataSet)   
    F = getCVsample(D, sampleID, k)    
    
    nTest, p = np.shape(F.E.X)
    n, s = np.shape(F.R.Y)
    n, p = np.shape(F.R.X)
    
    print('Training set size = ', n)
    print('         Test set = ', nTest)
    print(g)
    
    ''' L is the number of hidden hList + 1. (1 counts the input layer X) '''
    
    indx = [i for i in range(n)]
    random.shuffle(indx)
    
    xSample, ySample, labelsSample = fn.batchSample(batSize, 0, indx, 0, F.R)
    batchProgress = batSize
    
    if k == 0:
        initialList = fn.initializeNN(g, xSample, fns, dfns, b)
        xList, hList, gradients, fPs, stepSizes, norms, vList, iList = initialList
    
        ePast = 1
        rmse_past = 0
        diff = 1
        iList = [0]*L
        acc = 0
    else:
        
        xList[L - 1] = xSample
        for r in range(nH,-1,-1 ):
            AH = fn.augment(xList[r], 1) * hList[r]
        yHat = AH
    it = 0    
    epoch = 0
    while it < 5000: 
        epoch = int(batSize*it/n)
        
            
        if it % 501 == 0 and a > 1E-6:
            a = a/2
            
        ''' Forward Propagation '''     
        xList, fPs, yHat = fn.forwardPropagation(xList, hList, fns, dfns, fPs)
        
        ''' Interim update '''    
        alpha = .9 - .4*np.exp((1- it)/1000)
        for r in range(L):
            iList[r] = alpha*vList[r] + hList[r]
        
        ''' Compute the Gradients '''
        dEdyhat = dEdyhatf(ySample, yHat)  
        
        gradients, penalty =  fn.gradientComputer(g, gradients, fPs, xList, iList, dEdyhat, L1, L2)
    
    
        ''' Compute the step sizes '''
        for r in range(L):
            norms[r], stepSizes[r] = fn.RMSProp(norms[r], gradients[r], a)
            
        ''' Update coefficients '''
        for r in range(L):
            vList[r] = np.multiply(alpha, vList[r]) - np.multiply(stepSizes[r], gradients[r])
            hList[r] += vList[r]
            
    
        ''' Evaluate progress '''    
        # Evaluate progress 
        obsAcc = testAcc(F.E, hList, fns)
        objFunction = sum([0.25* np.mean([x**2 for x in dEdyhat[:,i]]) for i in range(s)])

        obsAcc.append(objFunction)
        try:
            progress = [.9*progress[i] + .1*obsAcc[i] for i in range(len(progress))]
        except(NameError):
            progress = obsAcc
            
        
        print('\r'+str(k) + '/'+str(K)+' \t'+str(it)  
           +'\t r-sqr(1) = '+ str(round(progress[0],3)) 
           +'\t r-sqr(2) = '+ str(round(progress[1],3))
           +'\t obj = '+ str(round(progress[2],5)),end="")
            #print('CV \t', epoch,'\t', it, '\t acc = '+ eStr+'\t'+str(round(alpha,3)),' ' ,a)
        if fold == 0:
        
            #iterations[it] = acc
            try:
                iterations[it].extend(acc)
            except:
                iterations = acc

        batchProgress += batSize
        if batchProgress >= n:
            '''Drawing the next batch sample...'''
            random.shuffle(indx)
            xList, ySample, labelsSample = fn.batchSample(batSize, it, indx, xList, trainSet)
            batchProgress = batSize
        it += 1    
        
    ''' iterations complete '''
        
    rSqrList = fn.testAcc(testSet, hList, fns)
    cvDict[k] = [dim(F.E.Y)[0], rsq]   

    cvAcc = [sum([cvDict[i][0] * cvDict[i][1][i] for i in range(k)])/n for i in range(s)]    
    print('\nCV adjusted R-sqr = '+'  '+'  '.join([str(round(a,3)) for a in cvAcc]))

    string = str(round(cvAcc,5))
    print('\n',g,L1,L2,'\t acc= '+string)
    
    X = fn.augment(trainSet.X,1)
    beta = np.linalg.solve(X.T*X, X.T*trainSet.Y) 
    E = testSet.Y - fn.augment(testSet.X,1)*beta
    rsq = fn.rSqr(testSet.Y,E)    
    lrDict[fold] = rsq
    print('Linear regression adj R2 = ',rsq)

    