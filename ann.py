#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 01:19:18 2018

@author: julien nyambal
"""

import numpy as np
import matplotlib.pyplot as plt

#number to learn
#limit at 141
num_to_learn = [[0,0],[0,1],[1,0],[1,1]]
#parameters of the network
theta_1 = [0.2,0.01]
theta_2 = [0.01]
#bias and data
bias = [1,1,1,1]
x = num_to_learn
#expected output
y = [0,1,1,1]
#hidden one 
h_1 = np.dot(x,theta_1)
#hypothesis, predicted output
h = h_1 * theta_2
#learning rate
alpha = 0.001

#J = np.sum(0.5*(y - h)**2)

#error, and theta lists
error, steps = [], []
old_J = 0
diff = 0
i = 0
m= 2

while (True):
    #Computation of the hidden output
    h_1 = np.dot(x,theta_1)
    #Computation of the expected output
    h = h_1 * theta_2
    #Computation of the loss
    loss = (h - y)
    #Computation of the gradient. Given the Mean Sqaure Error Loss      
    gradient_1 = np.dot(loss,x) /m
    gradient_2 = np.dot(loss,h_1) /m
    #m is optional as we have only one data point
    theta_1 = theta_1 - alpha * gradient_1
    theta_2 = theta_2 - alpha * gradient_2
#   theta[0] = theta[0] - alpha * gradient[0]
#   theta[1] = theta[1] - alpha * gradient[1]
    #Computation of the error function, again, in this case m is useless
    J = np.sum(0.5 * (1./m) * loss**2)
    error.append(J)
    print "Error ",i,":", J
    steps.append(i)
    #threshold the error to stop at it when less or equal to 10^-4
    diff = np.abs(old_J - J)
    print "Difference: ", diff
    if (diff <= 0.000000000001):
        break
    old_J = J
    i = i + 1
#print "Error: ", error
print "Parameters: ", theta_1
print "Predicted Outcome", h
plt.plot(steps,error )
plt.show()


    