#!/usr/bin/env python2

from train import buildmodel
from num2verilog import nums2verilog1d as n2v
import numpy as np
from glob import glob

datadir = 'weights/'
outfile = 'params'

def np2hex(arr):
    bytes_ = map(ord, arr.tobytes())
    hex_ = map(lambda n: '%.2X' % n, bytes_)
    rawhex = ''.join(hex_)
    return rawhex

if __name__ == '__main__':

    weightdict = []

    M = buildmodel()
    for layer in M.layers:
        if 'conv' in layer.name:
            weights = np.load(datadir+layer.name+'.npy')
            layer.set_weights(weights)
        
            weights, biases = weights
            print np.shape(weights)
            weights = np.transpose(weights, (3, 2, 0, 1))

            weights *= 1024
            biases *= 1024
            weights = weights.astype(np.int32)
            biases = biases.astype(np.int32)
            weights = weights.flatten()
            
            weightdict.append((layer.name+'_bias', biases))
            weightdict.append((layer.name+'_weights', weights))

    f = open(outfile, 'w')
    for layername, nums in weightdict:
        num = layername.split('_')[1]
        name = layername.split('_')[2]+num
        S = n2v(nums, name)        
        f.write(S)
    
    f.close()
