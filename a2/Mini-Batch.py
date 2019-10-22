#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:50:27 2019
For reference:
https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/
@author: shiang
"""


import numpy as np
import matplotlib.pyplot as plt
import MLCourse.dataloader as dtl

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction, ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction, ytest), ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction, ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions, ytest) / np.sqrt(ytest.shape[0])

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def predict(xtest, weight):
    return xtest.dot(weight)

if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 1
    stepsize = 0.01
    
    trainset, testset = dtl.load_ctscan(trainsize,testsize)
    xtrain = trainset[0]
    ytrain = trainset[1]
    xtest = testset[0]
    ytest = testset[1]
    weight = np.random.randn(385)
    errors = np.zeros(500)