#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 01:19:18 2018

@author: julien nyambal
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, deriv=False):
    if (deriv==True):
        return (x*(1-x))
    return 1/(1 + np.exp(-x))

def tanh(x, deriv=False):
    numerator,denominator,tanh_value,deriv_tanh_value = 0,0,0,0
    if (deriv==True):
        numerator = (1 - np.exp(-2*x))**2
        denominator = (1 + np.exp(-2*x))**2
        deriv_tanh_value = 1 - (numerator/denominator)
        return deriv_tanh_value
    numerator = 1 - np.exp(-2*x)
    denominator = 1 + np.exp(-2*x)
    tanh_value = numerator/denominator
    return tanh_value    

#input data
data = np.array([[0,0,1],
                 [0,1,1],
                 [1,0,1],
                 [1,1,1]])
#output
y = np.array([[0],[1],[1],[0]])
    
num_features = data.shape[1]
num_hidden_nodes = 7
num_outputs = y.shape[1]
num_epochs = 60000
alpha = 0.1
loss,iterations = [],[]

#seed
#np.random.seed(0)

theta0 = 2*np.random.random((num_features,num_hidden_nodes)) - 1
theta1 = 2*np.random.random((num_hidden_nodes,num_outputs)) - 1

for i in xrange(num_epochs):
    #layers
    layer_0 = data
    layer_1 = tanh(np.dot(layer_0,theta0))
    layer_2 = tanh(np.dot(layer_1,theta1))

#backpropagation
    #update layer 2
    layer_2_loss = y - layer_2
    loss_ = np.mean(np.abs(layer_2_loss))
    if (i % 100) == 0:
        print "Error:", loss_
    layer_2_gradient = layer_2_loss * tanh(layer_2,True)
    
    #update layer 1
    layer_1_loss = np.dot(layer_2_gradient,theta1.T)
    layer_1_gradient = layer_1_loss * tanh(layer_1,True)
    
    theta1 += alpha * np.dot(layer_1.T,layer_2_gradient) 
    theta0 += alpha * np.dot(layer_0.T,layer_1_gradient) 
    
    loss.append(loss_)
    iterations.append(i)

print layer_2  

plt.plot(iterations,loss)
plt.show() 