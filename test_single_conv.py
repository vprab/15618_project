#!/usr/bin/env python2

from num2verilog import nums2verilog1d as n2v
import numpy as np
import keras
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

inp = Input(shape = (3, 3, 2))
net = Convolution2D(filters = 1, kernel_size = 3, activation = 'relu', padding = 'same')(inp)
M = Model(inp, net)

custom_weights = np.reshape(np.array([[1,2,3],[4,5,6],[7,8,9],
                                      [9,8,7],[6,5,4],[3,2,1]]), (3,3,2,1))
custom_bias = np.array([-936])

convlayer = M.layers[1]
assert convlayer.name == 'conv2d_1'
convlayer.set_weights((custom_weights, custom_bias))

test_input = np.array([[[[2,3],[4,5],[6,7]],
                        [[8,9],[10,11],[12,13]],
                        [[14,15],[16,17],[18,10]]]])

test_out = np.squeeze(M.predict(test_input))
print test_out

###now to get the verilog test data###
print np.shape(custom_weights)

weights = np.transpose(custom_weights, (3,2,0,1)).flatten().astype(np.int32)
biases = custom_bias.flatten().astype(np.int32)
img = np.transpose(test_input[0], (2,0,1)).flatten()

print n2v(biases, 'biases')
print n2v(weights, 'weights')
print n2v(img, 'img')
