#!/usr/bin/env python2
import os
import time
import numpy as np
import keras
from train import buildmodel, loaddata

os.environ['CUDA_VISIBLE_DEVICES'] = ''
#to test CPU

datadir = 'weights/'


def uncat(vec):
    return np.argmax(vec)

if __name__ == '__main__':
    M = buildmodel(True)
    for layer in M.layers:
        wb = np.load(datadir+layer.name+'.npy')
        if not len(wb):
            continue
        
        weights, biases = wb
        #do modification
        weights *= 1024
        biases *= 1024
        weights = weights.astype(np.int32)
        biases = biases.astype(np.int32)        
        #finish modification

        #weights = np.transpose(weights, (1,0,2,3))
        
        layer.set_weights((weights, biases))

    X, Y = loaddata(0)

    M.layers.pop()
    M.layers.pop()
    M.outputs = [M.layers[-1].output]
    M.layers[-1].outbound_nodes = []

    rounds = 1
    st = time.time()
    for i in range(rounds):
        pred = M.predict_on_batch(X)
    et = time.time()
    print et-st
    print len(X)
    print (et-st)/len(X)/rounds


'''
gpu benchmarks: 1 million images (32x32x3)\\
power usage: (92-24)W*16.0 s = 1088 joules\\
this is 1.1 mJ per image
16 microseconds per image    

cpu: 50k images
power usage: 40W*10.37s = 415 joules
this is 8.3 mJ per image
211 microseconds per image

fpga: ??
'''
